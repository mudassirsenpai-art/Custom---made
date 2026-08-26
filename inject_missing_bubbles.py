#!/usr/bin/env python3
"""
inject_missing_bubbles.py

Adds bubbles that YOLO/OSB detection MISSED, directly by editing the same
combined_translations.json that Pass 1 already produced. Pass 1 is NEVER
re-run.

WORKFLOW
--------
1. Run Pass 1 exactly as normal. It produces checkpoints/*.pkl and
   combined_translations.json as always.

2. For any bubble that was missed, append a new entry to the END of that
   page's "bubbles" list in combined_translations.json. A missing-bubble
   entry looks exactly like a normal one, except it has "bbox" instead of
   "bubble_id" (since a never-detected bubble has no id yet):

       {
         "bbox": [494, 216, 664, 367],
         "is_outside_text": true,
         "ocr_text": "CHAPTER11:...",
         "translation": "CHAPTER 11: SECRET JAGAH",
         "sfx": false
       }

   You (or a chat AI you give the page image + this JSON to) only need to
   find the page ("source_file") and append entries like this to that
   page's "bubbles" array. Nothing else in the file needs to change.

3. Run this script once, pointing at combined_translations.json and the
   checkpoints folder:

       python inject_missing_bubbles.py \
           --combined-json combined_translations.json \
           --checkpoint-dir checkpoints/

   For every "bbox"-only entry it finds, per page, it will:
     a. Load that page's .pkl checkpoint.
     b. Crop the region out of the checkpoint's cleaned image (still the
        original untouched pixels there, since detection never found
        this bubble).
     c. Build a rectangular mask and run it through the project's own
        `process_single_bubble` (the exact function Pass 1 itself uses)
        to measure fill color / glow / outline / text style.
     d. Flat-fill that mask on the checkpoint's cleaned image with the
        detected color (fast path, no Flux/SDXL re-inpaint).
     e. Append the new bubble into the checkpoint's `sorted_bubble_data`
        and `processed_bubbles_info`.
     f. Compute a bubble_id (same function Pass 1 uses) and write it back
        into that entry in combined_translations.json, replacing "bbox".
   Then it re-saves both the checkpoint(s) it touched and
   combined_translations.json, in place (.bak backups are kept).

4. Run Pass 2 exactly as normal, from the SAME (now-updated)
   combined_translations.json. Every injected bubble now has a bubble_id
   like any other, so Pass 2 renders it automatically - no extra steps.

Run this from inside your Custom---made project root, in the same
environment (venv) Pass 1 / Pass 2 already run in (needs torch, opencv,
etc. from that project).

CAVEATS
-------
- Rectangular mask, not a real segmentation - fine for OSB boxes, mostly
  fine for round bubbles too as long as the bbox is reasonably tight.
- Flat-color fill (fast path), not a full AI inpaint - can look slightly
  patchy over complex/colored art behind the missed bubble.
- Must be run before Pass 2 renders that page. If Pass 2 already ran for
  it, just re-run Pass 2 for that page afterward.
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(path: Path, payload: dict) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--combined-json", required=True,
        help="Path to combined_translations.json (edited to include your bbox-only missing-bubble entries)",
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing the page .pkl checkpoint files from Pass 1",
    )
    parser.add_argument("--thresholding-value", type=int, default=200)
    parser.add_argument("--use-otsu-threshold", action="store_true", default=True)
    parser.add_argument("--roi-shrink-px", type=int, default=5)
    parser.add_argument(
        "--json-report", default=None,
        help="Optional path to write a machine-readable {added, skipped, failures: [...]} "
             "report to, for callers (e.g. the bot) that need structured results instead "
             "of parsing stdout.",
    )
    args = parser.parse_args()

    try:
        from core.pipeline import compute_bubble_id
        from core.image.cleaning import process_single_bubble
        from core.image.image_utils import pil_to_cv2
    except ImportError as e:
        print(
            f"ERROR: could not import project modules ({e}).\n"
            "Run this script from inside your Custom---made project root, "
            "in the same environment (venv) you use for Pass 1 / Pass 2.",
            file=sys.stderr,
        )
        sys.exit(1)

    import cv2
    from PIL import Image

    combined_path = Path(args.combined_json)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not combined_path.is_file():
        print(f"ERROR: combined JSON not found: {combined_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint_dir.is_dir():
        print(f"ERROR: checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    pages = combined.get("pages", [])
    if not pages:
        print("ERROR: no 'pages' found in combined JSON.", file=sys.stderr)
        sys.exit(1)

    total_added = 0
    total_skipped = 0
    any_json_change = False
    failures = []  # [{"source_file": ..., "bbox": [...], "reason": "..."}]

    for page in pages:
        source_file = page.get("source_file")
        checkpoint_name = page.get("checkpoint_file") or (
            f"{Path(source_file).stem}.pkl" if source_file else None
        )
        bubbles = page.get("bubbles", [])

        # Entries needing injection: have "bbox" and no usable "bubble_id".
        pending = [
            (i, b) for i, b in enumerate(bubbles)
            if b.get("bbox") and not b.get("bubble_id")
        ]
        if not pending:
            continue

        if not checkpoint_name:
            reason = "no checkpoint_file/source_file to locate .pkl"
            print(f"SKIP page '{source_file}': {reason}.", file=sys.stderr)
            total_skipped += len(pending)
            for _, b in pending:
                failures.append({"source_file": source_file, "bbox": b.get("bbox"), "reason": reason})
            continue

        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.is_file():
            reason = f"checkpoint '{checkpoint_path.name}' not found"
            print(f"SKIP page '{source_file}': {reason}.", file=sys.stderr)
            total_skipped += len(pending)
            for _, b in pending:
                failures.append({"source_file": source_file, "bbox": b.get("bbox"), "reason": reason})
            continue

        payload = load_checkpoint(checkpoint_path)
        pil_cleaned_image = payload["pil_cleaned_image"]
        sorted_bubble_data = payload.get("sorted_bubble_data", [])
        processed_bubbles_info = payload.get("processed_bubbles_info", [])
        existing_ids = {b.get("bubble_id") for b in sorted_bubble_data}

        image_bgr = pil_to_cv2(pil_cleaned_image)
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img_gray.shape[:2]

        page_added = 0

        for idx, bubble_entry in pending:
            bbox_raw = bubble_entry.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                reason = f"bad bbox {bbox_raw} (must be [x1, y1, x2, y2])"
                print(f"SKIP '{source_file}' entry {idx}: {reason}", file=sys.stderr)
                total_skipped += 1
                failures.append({"source_file": source_file, "bbox": bbox_raw, "reason": reason})
                continue

            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox_raw)
            except (TypeError, ValueError):
                reason = f"bbox values must be numbers, got {bbox_raw}"
                print(f"SKIP '{source_file}' entry {idx}: {reason}", file=sys.stderr)
                total_skipped += 1
                failures.append({"source_file": source_file, "bbox": bbox_raw, "reason": reason})
                continue

            x1, x2 = sorted((max(0, x1), min(img_width, x2)))
            y1, y2 = sorted((max(0, y1), min(img_height, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                reason = f"bbox too small or out of bounds after clamping to {img_width}x{img_height}"
                print(f"SKIP '{source_file}' entry {idx}: {reason}", file=sys.stderr)
                total_skipped += 1
                failures.append({"source_file": source_file, "bbox": bbox_raw, "reason": reason})
                continue
            bbox = (x1, y1, x2, y2)

            is_outside_text = bool(bubble_entry.get("is_outside_text", False))
            bubble_id = compute_bubble_id(bbox, is_outside_text=is_outside_text)

            if bubble_id in existing_ids:
                # Already present (e.g. script re-run) - just wire the id in.
                bubble_entry["bubble_id"] = bubble_id
                bubble_entry.pop("bbox", None)
                any_json_change = True
                continue

            base_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            base_mask[y1:y2, x1:x2] = 255

            try:
                (
                    final_mask,
                    fill_color_bgr,
                    is_colored_bubble,
                    sample_color_bgr,
                    text_bbox,
                    text_color_bgr,
                    text_median_line_height,
                    glow_color_rgb,
                    glow_radius,
                    lettering_style,
                ) = process_single_bubble(
                    base_mask=base_mask,
                    img_gray=img_gray,
                    img_height=img_height,
                    img_width=img_width,
                    thresholding_value=args.thresholding_value,
                    use_otsu_threshold=args.use_otsu_threshold,
                    roi_shrink_px=max(0, min(10, args.roi_shrink_px)),
                    verbose=False,
                    detection_bbox=bbox,
                    is_sam=False,
                    classify_colored=False,
                    processing_scale=1.0,
                    image_bgr=image_bgr,
                )
            except Exception as e:
                reason = f"cleaning failed: {e}"
                print(f"SKIP '{source_file}' bbox {bbox}: {reason}", file=sys.stderr)
                total_skipped += 1
                failures.append({"source_file": source_file, "bbox": list(bbox), "reason": reason})
                continue

            paint_color = sample_color_bgr if sample_color_bgr else fill_color_bgr
            if image_bgr.shape[2] == 4:
                image_bgr[final_mask == 255, :3] = paint_color
            else:
                image_bgr[final_mask == 255] = paint_color

            sorted_bubble_data.append(
                {
                    "bbox": list(bbox),
                    "is_outside_text": is_outside_text,
                    "ocr_text": bubble_entry.get("ocr_text", ""),
                    "bubble_id": bubble_id,
                }
            )
            processed_bubbles_info.append(
                {
                    "mask": final_mask,
                    "base_mask": base_mask,
                    "color": paint_color,
                    "bbox": list(bbox),
                    "is_colored": is_colored_bubble,
                    "text_bbox": text_bbox,
                    "text_color_bgr": text_color_bgr,
                    "text_median_line_height": text_median_line_height,
                    "glow_color_rgb": glow_color_rgb,
                    "glow_radius": glow_radius,
                    "lettering_style": lettering_style,
                    "is_sam": False,
                    "inpainted": False,
                }
            )
            existing_ids.add(bubble_id)

            # Replace the bbox-only entry in the JSON with a proper bubble_id
            # entry - translation/sfx/ocr_text the human/AI already wrote
            # stay untouched.
            bubble_entry["bubble_id"] = bubble_id
            bubble_entry.pop("bbox", None)
            any_json_change = True

            page_added += 1
            total_added += 1
            print(f"  + [{source_file}] {bubble_id}  bbox={bbox}")

        if page_added:
            fixed_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            payload["pil_cleaned_image"] = Image.fromarray(fixed_rgb)
            payload["sorted_bubble_data"] = sorted_bubble_data
            payload["processed_bubbles_info"] = processed_bubbles_info
            save_checkpoint(checkpoint_path, payload)
            print(f"Updated checkpoint '{checkpoint_path.name}' (+{page_added} bubble(s)).")

    if any_json_change:
        backup_path = combined_path.with_suffix(combined_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(combined_path, backup_path)
        combined_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Updated '{combined_path.name}' in place (backup saved as '{backup_path.name}').")

    print(f"\nDone. Added {total_added} bubble(s), skipped {total_skipped}.")
    if total_added:
        print("Run Pass 2 normally from this same combined_translations.json now.")

    if args.json_report:
        report = {"added": total_added, "skipped": total_skipped, "failures": failures}
        Path(args.json_report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if failures:
        # Non-zero exit whenever anything was skipped, even if some bubbles
        # elsewhere were added successfully - callers (e.g. the bot) should
        # treat a partial run as "needs a fix", not silently proceed to
        # Pass 2 with some bubbles still un-rendered.
        sys.exit(1)


if __name__ == "__main__":
    main()

import functools
from pathlib import Path
from typing import Any

import gradio as gr

from core.text.text_processing import text_layout_control_interactivity
from utils.model_metadata import (
    FLUX_SDCPP_DIFFUSION_QUANTS,
    flux_sdcpp_text_encoder_default,
    flux_sdcpp_text_encoder_quants,
    flux_sdcpp_valid_text_encoder_quant,
    flux_valid_backend,
)

from . import callbacks, settings_manager, utils

_FLUX_BACKEND_CHOICES_KLEIN = [
    ("sd.cpp", "sdcpp"),
    ("SDNQ", "sdnq"),
]
_FLUX_BACKEND_CHOICES_KONTEXT = [
    ("sd.cpp", "sdcpp"),
    ("SDNQ", "sdnq"),
    ("Nunchaku (CUDA)", "nunchaku"),
]
_FLUX_BACKEND_CHOICES_DEFAULT = [("SDNQ", "sdnq")]


def _flux_backend_choices(method: str):
    if method in ("flux_klein_9b", "flux_klein_4b"):
        return _FLUX_BACKEND_CHOICES_KLEIN
    if method == "flux_kontext":
        return _FLUX_BACKEND_CHOICES_KONTEXT
    return _FLUX_BACKEND_CHOICES_DEFAULT


_ALPHABETICAL_LANGUAGES = [
    "Afrikaans",
    "Albanian",
    "Arabic",
    "Armenian",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Catalan",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Persian (Farsi)",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Latvian",
    "Lithuanian",
    "Malay",
    "Marathi",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Serbian (Cyrillic)",
    "Serbian (Latin)",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tamil",
    "Telugu",
    "Thai",
    "Filipino (Tagalog)",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Welsh",
]

SOURCE_LANGUAGES = [
    "Japanese",
    "Korean",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
] + [
    lang
    for lang in _ALPHABETICAL_LANGUAGES
    if lang
    not in ["Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)"]
]

TARGET_LANGUAGES = ["English"] + [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang != "English"
]

# Languages supported by PaddleOCR-VL-1.6 (53 of the 59 in _ALPHABETICAL_LANGUAGES)
_PADDLE_OCR_VL_UNSUPPORTED = frozenset(
    ["Armenian", "Georgian", "Gujarati", "Hebrew", "Kannada", "Punjabi"]
)
PADDLE_OCR_VL_LANGUAGES = [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang not in _PADDLE_OCR_VL_UNSUPPORTED
]

js_credits = """
function() {
    const footer = document.querySelector('footer');
    if (footer) {
        // Check if credits already exist
        if (footer.parentNode.querySelector('.mangatl-credits')) {
            return;
        }
        const newContent = document.createElement('div');
        newContent.className = 'mangatl-credits'; // Add a class for identification
        newContent.innerHTML = 'made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

        newContent.style.textAlign = 'center';
        newContent.style.paddingTop = '50px';
        newContent.style.color = 'lightgray';

        // Style the hyperlink
        const link = newContent.querySelector('a');
        if (link) {
            link.style.color = 'gray';
            link.style.textDecoration = 'underline';
        }

        footer.parentNode.insertBefore(newContent, footer);
    }
}
"""

js_status_fade = """
() => {
    // Find the specific config status element by its ID
    const statusElement = document.getElementById('config_status_message');  // Config status

    // Apply fade logic only to the config status element
    if (statusElement) {
        if (statusElement && statusElement.textContent.trim() !== "") {
            clearTimeout(statusElement.fadeTimer);
            clearTimeout(statusElement.resetTimer);

            statusElement.style.display = 'block';
            statusElement.style.transition = 'none';
            statusElement.style.opacity = '1';

            const fadeDelay = 3000;
            const fadeDuration = 1000;

            statusElement.fadeTimer = setTimeout(() => {
                statusElement.style.transition = `opacity ${fadeDuration}ms ease-out`;
                statusElement.style.opacity = '0';

                statusElement.resetTimer = setTimeout(() => {
                    statusElement.style.display = 'none';
                    statusElement.style.opacity = '1';
                    statusElement.style.transition = 'none';
                }, fadeDuration);

            }, fadeDelay);
        } else {
            // Ensure hidden if empty
            statusElement.style.display = 'none';
        }
    }
}
"""

js_refresh_button_reset = """
() => {
    setTimeout(() => {
        const refreshButton = document.querySelector('.config-refresh-button button');
         if (refreshButton) {
            refreshButton.textContent = 'Refresh Models / Fonts';
            refreshButton.disabled = false;
        }
    }, 100); // Small delay to ensure Gradio update cycle completes
}
"""

js_refresh_button_processing = """
() => {
    const refreshButton = document.querySelector('.config-refresh-button button');
    if (refreshButton) {
        refreshButton.textContent = 'Refreshing...';
        refreshButton.disabled = true;
    }
    return []; // Required for JS function input/output
}
"""


js_reset_status_height = """
() => {
    setTimeout(() => {
        const ids = ['#translator_status_message textarea', '#batch_status_message textarea'];
        ids.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) {
                el.style.height = '';
                el.style.removeProperty('height');
            }
        });
    }, 100);
}
"""


def create_layout(
    models_dir: Path, fonts_base_dir: Path, target_device: Any
) -> gr.Blocks:
    """Creates the Gradio UI layout and connects callbacks."""

    css_path = Path(__file__).with_name("style.css")

    with gr.Blocks(
        title="MangaTranslator", js=js_credits, css_paths=str(css_path)
    ) as app:
        gr.Markdown("# MangaTranslator")

        font_choices, initial_default_font = utils.get_available_font_packs(
            fonts_base_dir
        )
        saved_settings = settings_manager.get_saved_settings()

        saved_font_pack = saved_settings.get("font_pack")
        default_font = (
            saved_font_pack
            if saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )
        batch_saved_font_pack = saved_settings.get("batch_font_pack")
        batch_default_font = (
            batch_saved_font_pack
            if batch_saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )

        saved_osb_font_pack = saved_settings.get("outside_text_osb_font_pack", "")
        if saved_osb_font_pack not in ([""] + font_choices):
            saved_osb_font_pack = ""

        initial_provider = saved_settings.get(
            "provider", settings_manager.DEFAULT_SETTINGS["provider"]
        )
        initial_model_name = saved_settings.get("model_name")

        if initial_provider == "OpenRouter" or initial_provider == "OpenAI-Compatible":
            initial_models_choices = [initial_model_name] if initial_model_name else []
        else:
            initial_models_choices = settings_manager.PROVIDER_MODELS.get(
                initial_provider, []
            )

        saved_max_tokens = saved_settings.get("max_tokens")
        if saved_max_tokens is not None:
            initial_max_tokens = saved_max_tokens
        else:
            is_reasoning = utils.is_reasoning_model(
                initial_provider, initial_model_name
            )
            initial_max_tokens = 16384 if is_reasoning else 4096

        # Calculate initial max_tokens maximum based on provider/model
        initial_max_tokens_cap = utils.get_max_tokens_cap(
            initial_provider, initial_model_name
        )
        initial_max_tokens_maximum = (
            initial_max_tokens_cap if initial_max_tokens_cap is not None else 63488
        )

        # --- Define UI Components ---
        with gr.Tabs():
            with gr.TabItem("Translator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            show_download_button=False,
                            image_mode=None,
                            elem_id="translator_input_image",
                        )
                        font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=default_font,
                            filterable=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            original_language_state = gr.State(
                                value=saved_settings.get("input_language", "Japanese")
                            )
                            input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get("input_language", "Japanese"),
                                allow_custom_value=True,
                            )
                            output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get("output_language", "English"),
                                allow_custom_value=True,
                            )
                        special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("special_instructions", ""),
                            lines=1,
                            max_lines=10,
                            elem_id="translator_special_instructions",
                        )
                        gr.Markdown(
                            "*SFX detection inside Outside-Text-Boxes (OSB) is automatic - "
                            "the model tags sound effects itself so they can be skipped "
                            "during inpainting. You don't need to type anything here for it.*",
                            elem_id="translator_sfx_note",
                        )
                        overlap_llm_with_inpaint = gr.Checkbox(
                            label="Overlap LLM With Inpainting",
                            value=bool(
                                saved_settings.get("overlap_llm_with_inpaint", False)
                            ),
                            info="Run LLM translation concurrently with inpainting.",
                        )
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            type="pil",
                            label="Translated Image",
                            interactive=False,
                            elem_id="translator_output_image",
                        )
                        status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="translator_status_message",
                        )
                        with gr.Row():
                            translate_button = gr.Button("Translate", variant="primary")
                            clear_button = gr.Button("Clear")
                            cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )

            with gr.TabItem("Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.File(
                            label="Upload Images or Folder",
                            file_count="directory",
                            file_types=["image"],
                            type="filepath",
                        )
                        input_zip = gr.File(
                            label="Upload ZIP Archive (preserves directory structure) or failed_paths.txt",
                            file_count="single",
                            file_types=[".zip", ".txt"],
                            type="filepath",
                        )
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=batch_default_font,
                            filterable=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            batch_original_language_state = gr.State(
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                )
                            )
                            batch_input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                ),
                                allow_custom_value=True,
                            )
                            batch_output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get(
                                    "batch_output_language", "English"
                                ),
                                allow_custom_value=True,
                            )
                        batch_special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("batch_special_instructions", ""),
                            lines=1,
                            max_lines=10,
                            elem_id="batch_special_instructions",
                        )
                        gr.Markdown(
                            "*SFX detection inside Outside-Text-Boxes (OSB) is automatic - "
                            "the model tags sound effects itself so they can be skipped "
                            "during inpainting. You don't need to type anything here for it.*",
                            elem_id="batch_sfx_note",
                        )
                        batch_parallel_requests = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=int(saved_settings.get("batch_parallel_requests", 1)),
                            step=1,
                            label="Parallel Requests",
                            info=(
                                "Controls the number of parallel workers, "
                                "one per page. Does not affect translation "
                                "quality."
                            ),
                        )
                        batch_parallel_within_pages = gr.Checkbox(
                            label="Use Parallel Requests Within Pages",
                            value=bool(
                                saved_settings.get("batch_parallel_within_pages", False)
                            ),
                            info=(
                                "Allows parallel workers to be shared with "
                                "independent Flux work within each page."
                            ),
                        )
                        batch_overlap_llm_with_inpaint = gr.Checkbox(
                            label="Overlap LLM With Inpainting",
                            value=bool(
                                saved_settings.get(
                                    "batch_overlap_llm_with_inpaint", False
                                )
                            ),
                            info="Run LLM translation concurrently with inpainting.",
                        )
                        batch_previous_context_image_count = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=int(
                                (
                                    saved_settings.get(
                                        "batch_previous_context_image_count", 0
                                    )
                                    if saved_settings.get(
                                        "send_full_page_context", True
                                    )
                                    else 0
                                )
                            ),
                            step=1,
                            label="Previous Context Images",
                            info=(
                                "Sends up to this many previous source pages "
                                "as visual reference when full-page context is enabled. "
                                "Might improve translation quality."
                            ),
                            interactive=(
                                saved_settings.get("send_full_page_context", True)
                                and saved_settings.get("ocr_method", "LLM") == "LLM"
                            ),
                        )
                        batch_previous_context_text_count = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=int(
                                saved_settings.get(
                                    "batch_previous_context_text_count", 3
                                )
                            ),
                            step=1,
                            label="Previous Context OCR Text",
                            info=(
                                "Sends up to this many previous pages' OCR text "
                                "transcripts as narrative reference. "
                                "Might improve translation quality."
                            ),
                        )
                        batch_retry_failed_once = gr.Checkbox(
                            label="Retry Failed Images Once",
                            value=bool(
                                saved_settings.get("batch_retry_failed_once", False)
                            ),
                            info=(
                                "Automatically retry each failed image once "
                                "at the very end."
                            ),
                        )
                    with gr.Column(scale=1):
                        batch_output_gallery = gr.Gallery(
                            label="Translated Images",
                            show_label=True,
                            columns=4,
                            height="auto",
                            object_fit="contain",
                        )
                        batch_status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="batch_status_message",
                        )
                        with gr.Row():
                            batch_process_button = gr.Button(
                                "Start Batch Translating", variant="primary"
                            )
                            batch_clear_button = gr.Button("Clear")
                            batch_cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )

            with gr.TabItem("Config", elem_id="settings-tab-container"):
                config_initial_provider = initial_provider
                config_initial_model_name = initial_model_name
                config_initial_models_choices = initial_models_choices

                with gr.Row(elem_id="config-button-row"):
                    save_config_btn = gr.Button(
                        "Save Config", variant="primary", scale=3
                    )
                    reset_defaults_btn = gr.Button(
                        "Reset Defaults", variant="secondary", scale=1
                    )

                # Assign specific ID for JS targeting
                config_status = gr.Markdown(elem_id="config_status_message")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="settings-nav"):
                        nav_buttons = []
                        setting_groups = []
                        nav_button_detection = gr.Button(
                            "Detection",
                            elem_classes=["nav-button", "nav-button-selected"],
                        )
                        nav_buttons.append(nav_button_detection)
                        nav_button_cleaning = gr.Button(
                            "Cleaning", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_cleaning)
                        nav_button_translation = gr.Button(
                            "Translation", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_translation)
                        nav_button_rendering = gr.Button(
                            "Rendering", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_rendering)
                        nav_button_outside_text = gr.Button(
                            "OSB Text", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_outside_text)
                        nav_button_output = gr.Button(
                            "Output", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_output)
                        nav_button_other = gr.Button("Other", elem_classes="nav-button")
                        nav_buttons.append(nav_button_other)

                    with gr.Column(scale=4, elem_id="config-content-area"):
                        # --- Detection Settings ---
                        with gr.Group(
                            visible=True, elem_classes="settings-group"
                        ) as group_detection:
                            gr.Markdown("### Speech Bubble Detection")
                            bubble_detector_model = gr.Radio(
                                choices=["yolo_1", "yolo_2", "yolo_3"],
                                value=saved_settings.get(
                                    "bubble_detector_model", "yolo_2"
                                ),
                                label="Bubble Detector Model",
                                info=(
                                    "Primary YOLO model for bubble detection. "
                                    "yolo_1/yolo_2: manga-trained. "
                                    "yolo_3: trained on manga, webtoon, manhua, "
                                    "and western comics - best for manhwa/manhua."
                                ),
                            )
                            confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("confidence", 0.6),
                                step=0.05,
                                label="Bubble Confidence Threshold",
                                info="Lower values detect more bubbles, but potentially include false positives.",
                            )
                            conjoined_detection_checkbox = gr.Checkbox(
                                value=saved_settings.get("conjoined_detection", True),
                                label="Conjoined Bubble Detection",
                                info=(
                                    "Uses a secondary RT-DETR model to detect and split "
                                    "conjoined speech bubbles into separate bubbles."
                                ),
                            )
                            conjoined_confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("conjoined_confidence", 0.35),
                                step=0.05,
                                label="Conjoined Bubble Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some conjoined bubbles.",
                                interactive=saved_settings.get(
                                    "conjoined_detection", True
                                ),
                            )
                            use_panel_sorting_checkbox = gr.Checkbox(
                                value=saved_settings.get("use_panel_sorting", True),
                                label="Use Panel-aware Sorting",
                                info=(
                                    "Use a panel detection YOLO model to group and sort speech bubbles "
                                    "within each panel for better reading order accuracy."
                                ),
                            )
                            panel_confidence = gr.Slider(
                                0.05,
                                1.0,
                                value=saved_settings.get("panel_confidence", 0.25),
                                step=0.05,
                                label="Panel Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some panels.",
                                interactive=saved_settings.get(
                                    "use_panel_sorting", True
                                ),
                            )
                            seg_model = gr.Radio(
                                choices=["sam3", "sam2", "yolo"],
                                value=saved_settings.get("seg_model", "yolo"),
                                label="Segmentation Model",
                                info=(
                                    "Model to use to segment speech bubbles. "
                                    "SAM 3 requires a HF token (shared with 'OSB Text' section)."
                                ),
                            )
                            osb_text_verification_checkbox = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_osb_text_verification", True
                                ),
                                label="Use AnimeText YOLO model for Bubble Verification",
                                info=(
                                    "Use the AnimeText YOLO model to confirm bubble detections fully cover text. "
                                    "Requires a Hugging Face token (shared with 'OSB Text' section)."
                                ),
                            )
                            config_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="Reading Direction",
                                value=saved_settings.get("reading_direction", "rtl"),
                                info="Order for sorting bubbles (rtl=Manga, ltr=Comic/Manhwa/Manhua).",
                                elem_id="config_reading_direction",
                            )
                        setting_groups.append(group_detection)

                        # --- Cleaning Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_cleaning:
                            gr.Markdown("### Mask Cleaning & Refinement")
                            thresholding_value = gr.Slider(
                                0,
                                255,
                                value=saved_settings.get("thresholding_value", 200),
                                step=1,
                                label="Fixed Threshold Value",
                                info=(
                                    "Brightness threshold for text detection. Lower helps clean "
                                    "edge-hugging text, but may thin bubble outlines"
                                ),
                                interactive=not saved_settings.get(
                                    "use_otsu_threshold", False
                                ),
                            )
                            use_otsu_threshold = gr.Checkbox(
                                value=saved_settings.get("use_otsu_threshold", False),
                                label="Force Automatic Thresholding (Otsu)",
                                info=(
                                    "Force Otsu's method for thresholding instead of the fixed value (on all bubbles). "
                                    "Recommended for varied lighting. Used as fallback when the fixed "
                                    "value fails, regardless of set value."
                                ),
                            )
                            roi_shrink_px = gr.Slider(
                                0,
                                10,
                                value=saved_settings.get("roi_shrink_px", 5),
                                step=1,
                                label="Shrink Threshold ROI (px)",
                                info=(
                                    "Shrink the threshold ROI inward by N pixels before fill. "
                                    "Lower helps clean edge-hugging text; higher preserves outlines."
                                ),
                            )

                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_translation:
                            gr.Markdown("### OCR & Translation")
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="Translation Mode",
                                value=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                ),
                                info=(
                                    "Determines whether to perform OCR and translation together or separately. "
                                    "'two-step' might improve translation quality for less-capable LLMs."
                                ),
                                elem_id="config_translation_mode",
                            )
                            initial_ocr_method = saved_settings.get(
                                "ocr_method",
                                settings_manager.DEFAULT_SETTINGS.get(
                                    "ocr_method", "LLM"
                                ),
                            )
                            ocr_method_radio = gr.Radio(
                                choices=[
                                    "LLM",
                                    "manga-ocr",
                                    "paddleocr-vl-1.6",
                                    "paddleocr-classic",
                                    "paddleocr-classic-v5",
                                ],
                                label="OCR Method",
                                value=initial_ocr_method,
                                info=(
                                    "Determines whether to use a vision-capable LLM or a local OCR model for OCR. "
                                    "Local OCR options enable text-only LLMs for translation "
                                    "and must be used in 'two-step' translation mode. "
                                    "'paddleocr-classic' (PP-OCRv4, not the VLM) "
                                    "supports Japanese/Korean/Chinese/English. "
                                    "'paddleocr-classic-v5' (PP-OCRv5, mobile) is newer "
                                    "and more accurate, covering the same languages."
                                ),
                                elem_id="ocr_method_radio",
                                interactive=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                )
                                != "one-step",
                            )

                            gr.Markdown("### LLM Settings")
                            available_providers = utils.get_available_providers(
                                initial_ocr_method
                            )
                            initial_provider_value = (
                                config_initial_provider
                                if config_initial_provider in available_providers
                                else (
                                    available_providers[0]
                                    if available_providers
                                    else "Google"
                                )
                            )
                            if initial_provider_value != config_initial_provider:
                                config_initial_provider = initial_provider_value
                            provider_selector = gr.Radio(
                                choices=available_providers,
                                label="Translation Provider",
                                value=initial_provider_value,
                                elem_id="provider_selector",
                            )
                            provider_state = gr.State(
                                value=initial_provider_value,
                            )
                            google_api_key = gr.Textbox(
                                label="Google AI Studio API Key",
                                placeholder="Enter Google AI Studio API key (starts with AI... / AQ...)",
                                type="password",
                                value=saved_settings.get("google_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Google"),
                                elem_id="google_api_key",
                                info="Stored locally. Or set via GOOGLE_API_KEY / GEMINI_API_KEY env var.",
                            )
                            openai_api_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="Enter OpenAI API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("openai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI"),
                                elem_id="openai_api_key",
                                info="Stored locally. Or set via OPENAI_API_KEY env var.",
                            )
                            anthropic_api_key = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="Enter Anthropic API key (starts with sk-ant-...)",
                                type="password",
                                value=saved_settings.get("anthropic_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Anthropic"),
                                elem_id="anthropic_api_key",
                                info="Stored locally. Or set via ANTHROPIC_API_KEY env var.",
                            )
                            xai_api_key = gr.Textbox(
                                label="SpaceXAI API Key",
                                placeholder="Enter SpaceXAI API key (starts with xai-...)",
                                type="password",
                                value=saved_settings.get("xai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "SpaceXAI"),
                                elem_id="xai_api_key",
                                info="Stored locally. Or set via SPACEXAI_API_KEY / XAI_API_KEY env var.",
                            )
                            deepseek_api_key = gr.Textbox(
                                label="DeepSeek API Key",
                                placeholder="Enter DeepSeek API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("deepseek_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "DeepSeek"),
                                elem_id="deepseek_api_key",
                                info="Stored locally. Or set via DEEPSEEK_API_KEY env var.",
                            )
                            zai_api_key = gr.Textbox(
                                label="Z.ai API Key",
                                placeholder="Enter Z.ai API key",
                                type="password",
                                value=saved_settings.get("zai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Z.ai"),
                                elem_id="zai_api_key",
                                info="Stored locally. Or set via ZAI_API_KEY env var.",
                            )
                            moonshot_api_key = gr.Textbox(
                                label="Moonshot API Key",
                                placeholder="Enter Moonshot API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("moonshot_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Moonshot AI"),
                                elem_id="moonshot_api_key",
                                info="Stored locally. Or set via MOONSHOT_API_KEY env var.",
                            )
                            mimo_api_key = gr.Textbox(
                                label="Xiaomi MiMo API Key",
                                placeholder="Enter MiMo API key (starts with sk-... or tp-...)",
                                type="password",
                                value=saved_settings.get("mimo_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Xiaomi MiMo"),
                                elem_id="mimo_api_key",
                                info=(
                                    "Stored locally. Or set via MIMO_API_KEY env var. "
                                    "Pay-as-you-go keys start with sk-; Token Plan keys start with tp-."
                                ),
                            )
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="Enter OpenRouter API key (starts with sk-or-...)",
                                type="password",
                                value=saved_settings.get("openrouter_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenRouter"),
                                elem_id="openrouter_api_key",
                                info="Stored locally. Or set via OPENROUTER_API_KEY env var.",
                            )
                            openai_compatible_url_input = gr.Textbox(
                                label="OpenAI-Compatible URL",
                                placeholder="Enter Base URL (e.g., http://localhost:8080/v1)",
                                type="text",
                                value=saved_settings.get(
                                    "openai_compatible_url",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "openai_compatible_url"
                                    ],
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_url_input",
                                info="Base URL of your OpenAI-Compatible API endpoint.",
                            )
                            openai_compatible_api_key_input = gr.Textbox(
                                label="OpenAI-Compatible API Key (Optional)",
                                placeholder="Enter API key if required",
                                type="password",
                                value=saved_settings.get(
                                    "openai_compatible_api_key", ""
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_api_key_input",
                                info="Stored locally. Or set via OPENAI_COMPATIBLE_API_KEY env var.",
                            )
                            config_model_name = gr.Dropdown(
                                choices=config_initial_models_choices,
                                label="Model",
                                value=config_initial_model_name,
                                info="Select the specific model for the chosen provider.",
                                elem_id="config_model_name",
                                allow_custom_value=True,
                            )
                            (
                                _initial_reasoning_effort_visible,
                                _initial_reasoning_effort_choices,
                                _initial_reasoning_effort_default,
                            ) = utils.get_reasoning_effort_config(
                                config_initial_provider, config_initial_model_name
                            )

                            _initial_reasoning_effort_value = saved_settings.get(
                                "reasoning_effort"
                            )
                            if _initial_reasoning_effort_value is None:
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif (
                                _initial_reasoning_effort_choices
                                and _initial_reasoning_effort_value
                                not in _initial_reasoning_effort_choices
                            ):
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif not _initial_reasoning_effort_choices:
                                _initial_reasoning_effort_value = None

                            _initial_reasoning_effort_info = (
                                utils.get_reasoning_effort_info_text(
                                    config_initial_provider,
                                    config_initial_model_name,
                                    _initial_reasoning_effort_choices,
                                )
                            )

                            _initial_reasoning_effort_label = (
                                utils.get_reasoning_effort_label(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            reasoning_effort_dropdown = gr.Radio(
                                choices=_initial_reasoning_effort_choices,
                                label=_initial_reasoning_effort_label,
                                value=_initial_reasoning_effort_value,
                                info=_initial_reasoning_effort_info,
                                visible=_initial_reasoning_effort_visible,
                                elem_id="reasoning_effort_dropdown",
                            )

                            # Effort dropdown (Claude Opus 4.5+ and Sonnet 4.6)
                            (
                                _initial_effort_visible,
                                _initial_effort_choices,
                                _initial_effort_default,
                            ) = utils.get_effort_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_effort_value = saved_settings.get("effort")
                            if _initial_effort_value is None:
                                _initial_effort_value = _initial_effort_default
                            elif (
                                _initial_effort_choices
                                and _initial_effort_value not in _initial_effort_choices
                            ):
                                _initial_effort_value = _initial_effort_default
                            elif not _initial_effort_choices:
                                _initial_effort_value = None

                            effort_dropdown = gr.Radio(
                                choices=_initial_effort_choices,
                                label="Effort",
                                value=_initial_effort_value,
                                info="Controls token spending eagerness. Opus 4.5+, Sonnet 4.6, Fable 5 only.",
                                visible=_initial_effort_visible,
                                elem_id="effort_dropdown",
                            )

                            # Verbosity dropdown (GPT-5 series only)
                            (
                                _initial_verbosity_visible,
                                _initial_verbosity_choices,
                                _initial_verbosity_default,
                            ) = utils.get_verbosity_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_verbosity_value = saved_settings.get("verbosity")
                            if _initial_verbosity_value is None:
                                _initial_verbosity_value = _initial_verbosity_default
                            elif (
                                _initial_verbosity_choices
                                and _initial_verbosity_value
                                not in _initial_verbosity_choices
                            ):
                                _initial_verbosity_value = _initial_verbosity_default
                            elif not _initial_verbosity_choices:
                                _initial_verbosity_value = None

                            verbosity_dropdown = gr.Radio(
                                choices=_initial_verbosity_choices,
                                label="Verbosity",
                                value=_initial_verbosity_value,
                                info="Controls response verbosity. GPT-5 series only.",
                                visible=_initial_verbosity_visible,
                                elem_id="verbosity_dropdown",
                            )

                            _initial_enable_web_search_visible = (
                                config_initial_provider
                                not in ("OpenAI-Compatible", "DeepSeek")
                            )
                            (
                                _initial_enable_web_search_label,
                                _initial_enable_web_search_info,
                            ) = utils.get_enable_web_search_label_and_info(
                                config_initial_provider
                                if _initial_enable_web_search_visible
                                else "Google"
                            )

                            enable_web_search_checkbox = gr.Checkbox(
                                label=_initial_enable_web_search_label,
                                value=saved_settings.get("enable_web_search", False),
                                info=_initial_enable_web_search_info,
                                visible=_initial_enable_web_search_visible,
                                elem_id="enable_web_search_checkbox",
                            )

                            # Compute initial visibility for enable_code_execution
                            _initial_enable_code_execution_visible = (
                                utils.is_code_execution_visible(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            enable_code_execution_checkbox = gr.Checkbox(
                                label="Enable Code Execution with Images",
                                value=saved_settings.get(
                                    "enable_code_execution", False
                                ),
                                info="Allow Gemini 3 Flash to zoom and inspect image details using code execution.",
                                visible=_initial_enable_code_execution_visible,
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl-1.6", "paddleocr-classic", "paddleocr-classic-v5"),
                                elem_id="enable_code_execution_checkbox",
                            )

                            (
                                _initial_image_detail_visible,
                                _initial_image_detail_choices,
                                _initial_image_detail_default,
                                _initial_image_detail_info,
                            ) = utils.get_image_detail_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_image_detail_value = saved_settings.get(
                                "image_detail", _initial_image_detail_default
                            )
                            if (
                                _initial_image_detail_choices
                                and _initial_image_detail_value
                                not in _initial_image_detail_choices
                            ):
                                _initial_image_detail_value = (
                                    _initial_image_detail_default
                                )

                            image_detail_dropdown = gr.Radio(
                                label="Image Detail",
                                choices=_initial_image_detail_choices,
                                value=_initial_image_detail_value,
                                info=_initial_image_detail_info,
                                visible=_initial_image_detail_visible,
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl-1.6", "paddleocr-classic", "paddleocr-classic-v5"),
                                elem_id="image_detail_dropdown",
                            )

                            # Compute initial visibility for media_resolution (Google/SpaceXAI providers only)
                            _mr_bubbles_visible_init, _, _ = (
                                utils.get_media_resolution_config(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )
                            _initial_media_resolution_visible = (
                                config_initial_provider == "Google"
                                and not _mr_bubbles_visible_init
                            )
                            initial_media_resolution_value = saved_settings.get(
                                "media_resolution", "auto"
                            )

                            media_resolution_dropdown = gr.Radio(
                                label="Media Resolution",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_value,
                                info="Resolution for Gemini to process bubble/context images.",
                                visible=_initial_media_resolution_visible,
                                elem_id="media_resolution_dropdown",
                            )

                            # Compute initial visibility for Gemini 3 and SpaceXAI specific media resolution options
                            _mr_bubbles_visible, _mr_choices, _mr_info_base = (
                                utils.get_media_resolution_config(
                                    config_initial_provider, config_initial_model_name
                                )
                            )
                            _mr_bubbles_info = _mr_info_base.replace(
                                "process images", "process bubble images"
                            )
                            _mr_context_info = _mr_info_base.replace(
                                "process images", "process context (full page) images"
                            )

                            initial_media_resolution_bubbles_value = saved_settings.get(
                                "media_resolution_bubbles", "auto"
                            )
                            initial_media_resolution_context_value = saved_settings.get(
                                "media_resolution_context", "auto"
                            )

                            media_resolution_bubbles_dropdown = gr.Radio(
                                label="Media Resolution (Bubbles)",
                                choices=_mr_choices,
                                value=initial_media_resolution_bubbles_value,
                                info=_mr_bubbles_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_bubbles_dropdown",
                            )

                            media_resolution_context_dropdown = gr.Radio(
                                label="Media Resolution (Context)",
                                choices=_mr_choices,
                                value=initial_media_resolution_context_value,
                                info=_mr_context_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_context_dropdown",
                            )

                            initial_use_custom_sampling_value = saved_settings.get(
                                "use_custom_sampling", True
                            )
                            initial_reasoning_effort = saved_settings.get(
                                "reasoning_effort"
                            )
                            (
                                initial_temp_interactive,
                                initial_top_p_interactive,
                                initial_top_k_interactive,
                            ) = utils.get_sampling_slider_interactivity(
                                config_initial_provider,
                                config_initial_model_name,
                                initial_reasoning_effort,
                                use_custom_sampling=initial_use_custom_sampling_value,
                            )
                            initial_use_custom_sampling_visible = (
                                utils.is_use_custom_sampling_visible(
                                    config_initial_provider,
                                    config_initial_model_name,
                                    initial_reasoning_effort,
                                )
                            )

                            use_custom_sampling_checkbox = gr.Checkbox(
                                label="Use Custom Sampling Parameters",
                                value=initial_use_custom_sampling_value,
                                info=(
                                    "Send custom temperature, top-p, and top-k values. "
                                    "Disabling may improve translation quality for some models."
                                ),
                                visible=initial_use_custom_sampling_visible,
                                elem_id="use_custom_sampling_checkbox",
                            )
                            temperature = gr.Slider(
                                0,
                                2.0,
                                value=saved_settings.get("temperature", 0.1),
                                step=0.05,
                                label="Temperature",
                                info="Controls creativity. Lower = deterministic; higher = random.",
                                interactive=initial_temp_interactive,
                                elem_id="config_temperature",
                            )
                            top_p = gr.Slider(
                                0,
                                1,
                                value=saved_settings.get("top_p", 0.95),
                                step=0.05,
                                label="Top P",
                                info="Controls diversity. Lower = focused; higher = random.",
                                interactive=initial_top_p_interactive,
                                elem_id="config_top_p",
                            )
                            top_k = gr.Slider(
                                0,
                                64,
                                value=saved_settings.get("top_k", 64),
                                step=1,
                                label="Top K",
                                info="Limits sampling pool to top K tokens.",
                                interactive=initial_top_k_interactive,
                                elem_id="config_top_k",
                            )
                            max_tokens = gr.Slider(
                                2048,
                                initial_max_tokens_maximum,
                                value=initial_max_tokens,
                                step=1024,
                                label="Max Tokens",
                                info="Maximum number of tokens in the response.",
                                elem_id="config_max_tokens",
                            )

                            gr.Markdown("### Context & Upscaling")
                            send_full_page_context = gr.Checkbox(
                                value=saved_settings.get(
                                    "send_full_page_context", True
                                ),
                                label="Send Full Page to LLM",
                                info=(
                                    "Include full page image as context. Might improve translation quality. "
                                    "Disable if refusals/using less-capable models or to reduce token usage."
                                ),
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl-1.6", "paddleocr-classic", "paddleocr-classic-v5"),
                            )
                            whiteout_conjoined_bubbles = gr.Checkbox(
                                value=saved_settings.get(
                                    "whiteout_conjoined_bubbles", True
                                ),
                                label="White-out Conjoined Bubbles",
                                info=(
                                    "White-out text from neighboring conjoined bubbles to avoid translating "
                                    "the same text multiple times. Disable if encountering issues."
                                ),
                            )
                            upscale_method = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                    ("LANCZOS", "lanczos"),
                                    ("None", "none"),
                                ],
                                value=saved_settings.get(
                                    "upscale_method", "model_lite"
                                ),
                                label="Bubble/Context Resizing Method",
                                info=(
                                    "Method to resize cropped bubble images/full page before sending to LLM/OCR model. "
                                    "Model is best quality, Model (Lite) is slightly worse quality but faster/less "
                                    "memory, LANCZOS is worst quality but fastest/least memory."
                                ),
                            )
                            initial_upscale_method = saved_settings.get(
                                "upscale_method", "model_lite"
                            )
                            sliders_interactive = initial_upscale_method != "none"
                            bubble_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("bubble_min_side_pixels", 128),
                                step=16,
                                label="Bubble Min Side Pixels",
                                info=(
                                    "Target minimum side length for speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_bubble_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                            context_image_max_side_pixels = gr.Slider(
                                512,
                                2560,
                                value=saved_settings.get(
                                    "context_image_max_side_pixels", 1024
                                ),
                                step=128,
                                label="Context Image Max Side Pixels",
                                info=(
                                    "Target maximum side length for full page image resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_context_image_max_side_pixels",
                                interactive=sliders_interactive,
                            )
                            osb_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("osb_min_side_pixels", 128),
                                step=16,
                                label="OSB Text Min Side Pixels",
                                info=(
                                    "Target minimum side length for outside speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_osb_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                        setting_groups.append(group_translation)

                        # --- Rendering Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_rendering:
                            gr.Markdown("### Font Rendering")
                            max_font_size = gr.Slider(
                                5,
                                50,
                                value=saved_settings.get("max_font_size", 16),
                                step=1,
                                label="Max Font Size (px)",
                                info="The largest font size the renderer will attempt to use.",
                            )
                            min_font_size = gr.Slider(
                                5,
                                50,
                                value=saved_settings.get("min_font_size", 8),
                                step=1,
                                label="Min Font Size (px)",
                                info="The smallest font size the renderer will attempt to use before giving up.",
                            )
                            line_spacing_mult = gr.Slider(
                                0.5,
                                2.0,
                                value=saved_settings.get("line_spacing_mult", 1.0),
                                step=0.05,
                                label="Line Spacing Multiplier",
                                info="Adjusts the vertical space between lines of text (1.0 = standard).",
                            )
                            use_subpixel_rendering = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_subpixel_rendering", False
                                ),
                                label="Use Subpixel Rendering",
                                info=(
                                    "Improves text clarity on RGB-based displays. "
                                    "Disable if using a PenTile-based display (i.e., an OLED screen)"
                                ),
                            )
                            font_hinting = gr.Radio(
                                choices=["none", "slight", "normal", "full"],
                                value=saved_settings.get("font_hinting", "none"),
                                label="Font Hinting",
                                info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for "
                                "high-res displays.",
                            )
                            use_ligatures = gr.Checkbox(
                                value=saved_settings.get("use_ligatures", False),
                                label="Use Standard Ligatures (e.g., fi, fl)",
                                info="Enables common letter combinations to be rendered as single glyphs "
                                "(must be supported by the font).",
                            )
                            gr.Markdown("### Text Layout")
                            detach_trailing_punctuation = gr.Checkbox(
                                value=saved_settings.get(
                                    "detach_trailing_punctuation", True
                                ),
                                label="Detach Trailing Punctuation",
                                info=(
                                    "Move trailing punctuation clusters onto a new line "
                                    "for better text wrapping."
                                ),
                            )
                            auto_vertical_text = gr.Checkbox(
                                value=saved_settings.get("auto_vertical_text", False),
                                label="Auto Vertical Text for Tall Bubbles",
                                info=(
                                    "Stack short translated text vertically in tall speech bubbles "
                                    "when it improves readability."
                                ),
                            )
                            _saved_hyphenate = saved_settings.get(
                                "hyphenate_before_scaling", True
                            )
                            _text_layout_flags = text_layout_control_interactivity(
                                saved_settings.get("output_language", "English"),
                                saved_settings.get("batch_output_language", "English"),
                                _saved_hyphenate,
                            )
                            # Force off when language-irrelevant: checkboxes do not
                            # visually grey out from interactive=False alone.
                            _hyphenate_relevant = _text_layout_flags[
                                "hyphenate_before_scaling"
                            ]
                            _effective_hyphenate = (
                                _saved_hyphenate if _hyphenate_relevant else False
                            )
                            _text_layout_flags = text_layout_control_interactivity(
                                saved_settings.get("output_language", "English"),
                                saved_settings.get("batch_output_language", "English"),
                                _effective_hyphenate,
                            )
                            hyphenate_before_scaling = gr.Checkbox(
                                value=_effective_hyphenate,
                                label="Hyphenate Long Words",
                                info="Try inserting hyphens when wrapping before reducing font size.",
                                interactive=_hyphenate_relevant,
                            )
                            hyphen_penalty = gr.Slider(
                                100,
                                2000,
                                value=saved_settings.get("hyphen_penalty", 1000.0),
                                step=50,
                                label="Hyphen Penalty",
                                info="Penalty for hyphenated line breaks in text layout. "
                                "Increase to discourage hyphenation.",
                                interactive=_text_layout_flags["hyphen_penalty"],
                            )
                            hyphenation_min_word_length = gr.Slider(
                                4,
                                10,
                                value=saved_settings.get(
                                    "hyphenation_min_word_length", 8
                                ),
                                step=1,
                                label="Min Word Length for Hyphenation",
                                info="Minimum word length required for hyphenation.",
                                interactive=_text_layout_flags[
                                    "hyphenation_min_word_length"
                                ],
                            )
                            badness_exponent = gr.Slider(
                                2.0,
                                4.0,
                                value=saved_settings.get("badness_exponent", 3.0),
                                step=0.5,
                                label="Badness Exponent",
                                info="Exponent for line badness calculation in text layout. "
                                "Increase to avoid loose lines.",
                            )
                            padding_pixels = gr.Slider(
                                2,
                                12,
                                value=saved_settings.get("padding_pixels", 4.0),
                                step=1,
                                label="Padding Pixels",
                                info="Padding between text and the edge of the speech bubble. "
                                "Increase for more space between text and bubble boundaries.",
                            )
                            supersampling_factor = gr.Slider(
                                1,
                                16,
                                value=saved_settings.get("supersampling_factor", 4),
                                step=1,
                                label="Supersampling Factor",
                                info="Render text at Nx resolution then downscale for smoother edges. "
                                "Higher values improve quality but use slightly more memory. 1 = disabled.",
                            )
                        setting_groups.append(group_rendering)

                        # --- Outside Text Removal Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_outside_text:
                            gr.Markdown("### Outside Speech Bubble Text")
                            outside_text_huggingface_token = gr.Textbox(
                                value=saved_settings.get(
                                    "outside_text_huggingface_token", ""
                                ),
                                label="HuggingFace Token (Required for certain features)",
                                type="password",
                                info=(
                                    "Required for downloading AnimeText YOLO (required for OSB detection), "
                                    "Flux.1 Kontext Nunchaku, and/or SAM 3 models from HuggingFace Hub. "
                                    "Can also set via HF_TOKEN env var."
                                ),
                            )
                            outside_text_enabled = gr.Checkbox(
                                value=saved_settings.get("outside_text_enabled", False),
                                label="Enable OSB Text Detection",
                                info="Detect, inpaint, and translate text outside speech bubbles.",
                            )

                            # Wrap all settings except the enable checkbox and token in a Column with visibility control
                            with gr.Column(
                                visible=saved_settings.get(
                                    "outside_text_enabled", False
                                )
                            ) as outside_text_settings_wrapper:
                                gr.Markdown("### Detection")
                                outside_text_osb_confidence = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_osb_confidence", 0.6
                                    ),
                                    step=0.05,
                                    label="OSB Text Detection Confidence",
                                    info="Lower values detect more text, but potentially include false positives.",
                                )
                                outside_text_bbox_expansion_percent = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_bbox_expansion_percent", 0.1
                                    ),
                                    step=0.05,
                                    label="Bounding Box Expansion",
                                    info=(
                                        "Percentage to expand bounding boxes for text detection. "
                                        "Higher values capture more context around text."
                                    ),
                                )
                                outside_text_text_box_proximity_ratio = gr.Slider(
                                    0.01,
                                    0.1,
                                    value=saved_settings.get(
                                        "outside_text_text_box_proximity_ratio", 0.02
                                    ),
                                    step=0.01,
                                    label="Text Box Proximity Ratio",
                                    info=(
                                        "Ratio for grouping nearby text boxes (as fraction of image dimension). "
                                        "Increase to group more distant boxes together."
                                    ),
                                )
                                outside_text_min_area_ignore_ratio_percent = gr.Slider(
                                    0.0,
                                    5.0,
                                    value=saved_settings.get(
                                        "outside_text_min_area_ignore_ratio", 0.0
                                    )
                                    * 100.0,
                                    step=0.1,
                                    label="Ignore Regions Below Area (%)",
                                    info=(
                                        "Skip OSB regions whose bounding-box area is below this "
                                        "percentage of the image."
                                    ),
                                )
                                outside_text_enable_page_number_filtering = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                    label="Filter Page Numbers",
                                    info=(
                                        "Use manga-ocr on margin detections to drop likely page numbers. "
                                        "Slightly slower and may detect false positives."
                                    ),
                                )
                                outside_text_page_filter_margin_threshold = gr.Slider(
                                    0.0,
                                    0.3,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_margin_threshold",
                                        0.1,
                                    ),
                                    step=0.01,
                                    label="Page Number Margin Ratio",
                                    info=(
                                        "Maximum vertical margin (ratio of height) for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                outside_text_page_filter_min_area_ratio = gr.Slider(
                                    0.0,
                                    0.2,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_min_area_ratio",
                                        0.05,
                                    ),
                                    step=0.01,
                                    label="Page Number Min Area Ratio",
                                    info=(
                                        "Minimum area ratio for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                gr.Markdown("### Inpainting")
                                outside_text_inpainting_method = gr.Radio(
                                    value=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    ),
                                    choices=[
                                        ("Flux.2 Klein 9B", "flux_klein_9b"),
                                        ("Flux.2 Klein 4B", "flux_klein_4b"),
                                        ("Flux.1 Kontext (12B)", "flux_kontext"),
                                        ("LaMa", "lama"),
                                        ("LaMa-Large", "lama_large"),
                                        ("OpenCV", "opencv"),
                                        ("None (text background)", "none"),
                                    ],
                                    label="Inpainting Method",
                                    info=(
                                        "Klein models are newer, but may introduce minor color "
                                        "shifts. Kontext does not shift colors, but is more dated. "
                                        "LaMa/LaMa-Large are fast CPU-friendly CNN inpainters "
                                        "(LaMa-Large is a higher-capacity checkpoint, better on "
                                        "complex textures but slower)."
                                    ),
                                )
                                _initial_method = saved_settings.get(
                                    "outside_text_inpainting_method", "flux_klein_4b"
                                )
                                _is_kontext = _initial_method == "flux_kontext"
                                _is_klein_model = _initial_method in (
                                    "flux_klein_9b",
                                    "flux_klein_4b",
                                )
                                _backend_visible = _is_klein_model or _is_kontext
                                _initial_backend = flux_valid_backend(
                                    _initial_method,
                                    saved_settings.get(
                                        "outside_text_flux_backend", "sdnq"
                                    ),
                                )
                                outside_text_flux_backend_state = gr.State(
                                    _initial_backend
                                )
                                outside_text_flux_backend = gr.Radio(
                                    choices=_flux_backend_choices(_initial_method),
                                    value=_initial_backend,
                                    label="Flux Backend",
                                    info=(
                                        "SDNQ/sd.cpp: cross-platform, no HF token. "
                                        "Nunchaku: CUDA-only, HF token required."
                                    ),
                                    visible=_backend_visible,
                                )
                                outside_text_flux_residual_diff_threshold = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_flux_residual_diff_threshold",
                                        0.15,
                                    ),
                                    step=0.01,
                                    label="Residual Diff Threshold",
                                    info=(
                                        "First Block Caching threshold for Kontext via Nunchaku. "
                                        "Higher = faster, but lower quality."
                                    ),
                                    interactive=(
                                        _is_kontext and _initial_backend == "nunchaku"
                                    ),
                                    visible=(
                                        _is_kontext and _initial_backend == "nunchaku"
                                    ),
                                )
                                _show_low_vram = (
                                    _is_klein_model or _is_kontext
                                ) and _initial_backend == "sdnq"
                                _show_sdcpp_quant = (
                                    _is_klein_model or _is_kontext
                                ) and _initial_backend == "sdcpp"
                                outside_text_flux_low_vram = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_low_vram", False
                                    ),
                                    label="Low VRAM Mode",
                                    info="Sequential CPU offload for SDNQ.",
                                    visible=_show_low_vram,
                                )
                                outside_text_flux_sdcpp_cache_mode = gr.Radio(
                                    choices=[
                                        ("Spectrum", "spectrum"),
                                        ("Cache-DiT", "cache-dit"),
                                        ("TaylorSeer", "taylorseer"),
                                        ("DBCache", "dbcache"),
                                        ("None", "none"),
                                    ],
                                    value=saved_settings.get(
                                        "outside_text_flux_sdcpp_cache_mode",
                                        "none",
                                    ),
                                    label="Cache Method",
                                    info=(
                                        "Caching methods for Flux via sd.cpp. Ordered fastest/worst quality -> slowest/best quality. "
                                        "Warmup is 25% of the selected step count."
                                    ),
                                    visible=(
                                        (_is_klein_model or _is_kontext)
                                        and _initial_backend == "sdcpp"
                                    ),
                                )
                                _saved_diffusion_quant = saved_settings.get(
                                    "outside_text_flux_sdcpp_diffusion_quant",
                                    "Q4_K_M",
                                )
                                _initial_diffusion_quant = (
                                    _saved_diffusion_quant
                                    if _saved_diffusion_quant
                                    in FLUX_SDCPP_DIFFUSION_QUANTS
                                    else "Q4_K_M"
                                )
                                outside_text_flux_sdcpp_diffusion_quant = gr.Radio(
                                    choices=[
                                        (quant, quant)
                                        for quant in FLUX_SDCPP_DIFFUSION_QUANTS
                                    ],
                                    value=_initial_diffusion_quant,
                                    label="Flux Model Quant",
                                    info=(
                                        "Quantization level for the model. Ordered from "
                                        "largest/best quality to smallest/lowest quality."
                                    ),
                                    visible=_show_sdcpp_quant,
                                )
                                outside_text_flux_sdcpp_diffusion_quant_state = (
                                    gr.State(_initial_diffusion_quant)
                                )
                                _available_text_encoder_quants = (
                                    flux_sdcpp_text_encoder_quants(_initial_method)
                                )
                                _text_encoder_quants = (
                                    _available_text_encoder_quants
                                    or flux_sdcpp_text_encoder_quants("flux_klein_4b")
                                )
                                _text_encoder_default = flux_sdcpp_text_encoder_default(
                                    _initial_method
                                )
                                _saved_text_encoder_quant = saved_settings.get(
                                    "outside_text_flux_sdcpp_text_encoder_quant",
                                    _text_encoder_default,
                                )
                                _initial_text_encoder_quant = (
                                    flux_sdcpp_valid_text_encoder_quant(
                                        _initial_method, _saved_text_encoder_quant
                                    )
                                )
                                outside_text_flux_sdcpp_text_encoder_quant = gr.Radio(
                                    choices=[
                                        (quant, quant) for quant in _text_encoder_quants
                                    ],
                                    value=_initial_text_encoder_quant,
                                    label="Text Encoder Model Quant",
                                    info=(
                                        "Quantization level for the model. Ordered from "
                                        "largest/best quality to smallest/lowest quality."
                                    ),
                                    visible=(
                                        _show_sdcpp_quant
                                        and bool(_available_text_encoder_quants)
                                    ),
                                )
                                outside_text_flux_sdcpp_text_encoder_quant_state = (
                                    gr.State(_initial_text_encoder_quant)
                                )
                                outside_text_flux_num_inference_steps = gr.Slider(
                                    1,
                                    (
                                        30
                                        if saved_settings.get(
                                            "outside_text_inpainting_method",
                                            "flux_klein_4b",
                                        )
                                        == "flux_kontext"
                                        else 12
                                    ),
                                    value=saved_settings.get(
                                        "outside_text_flux_num_inference_steps", 4
                                    ),
                                    step=1,
                                    label="Steps",
                                    info=(
                                        "Klein: 4 is recommended. "
                                        "Kontext: 6-15 is recommended."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    != "opencv",
                                )
                                _is_klein_for_lum = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) in ("flux_klein_9b", "flux_klein_4b")
                                _is_flux_for_klein_options = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) not in ("opencv", "none")
                                _upscale_small_crops_enabled = saved_settings.get(
                                    "outside_text_flux_upscale_small_crops", True
                                )
                                _group_flux_regions_enabled = saved_settings.get(
                                    "outside_text_flux_group_regions", False
                                )
                                outside_text_flux_luminance_correction = gr.Checkbox(
                                    value=(
                                        saved_settings.get(
                                            "outside_text_flux_luminance_correction",
                                            True,
                                        )
                                        if not _is_klein_for_lum
                                        else (
                                            saved_settings.get(
                                                "outside_text_flux_luminance_correction",
                                                True,
                                            )
                                            if (
                                                _upscale_small_crops_enabled
                                                and not _group_flux_regions_enabled
                                            )
                                            else False
                                        )
                                    ),
                                    label="Luminance Correction",
                                    info=(
                                        "Try and match generated patch brightness to surrounding context. "
                                        "Only available for ungrouped Klein crops scaled to ~1MP."
                                    ),
                                    visible=_is_klein_for_lum,
                                    interactive=(
                                        _is_klein_for_lum
                                        and _upscale_small_crops_enabled
                                        and not _group_flux_regions_enabled
                                    ),
                                )
                                outside_text_flux_upscale_small_crops = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_upscale_small_crops", True
                                    ),
                                    label="Upscale Klein Crops to ~1MP",
                                    info=(
                                        "Scale small Klein inpaint crops before Flux inference for better cleanup."
                                    ),
                                    visible=_is_flux_for_klein_options,
                                    interactive=_is_klein_for_lum,
                                )
                                outside_text_flux_group_regions = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_group_regions", False
                                    ),
                                    label="Group Flux Regions",
                                    info=(
                                        "Run one Flux pass over a combined expanded mask for all non-solid OSB regions."
                                    ),
                                    visible=_is_flux_for_klein_options,
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )
                                outside_text_seed = gr.Number(
                                    value=saved_settings.get("outside_text_seed", 1),
                                    label="Seed",
                                    info="Seed for reproducible inpainting (-1 = random)",
                                    precision=0,
                                    interactive=saved_settings.get(
                       

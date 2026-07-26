"""Side-effect-light registration for native Nocturne services."""

import logging
from threading import Lock

LOGGER = logging.getLogger("nocturne")

_registration_lock = Lock()
_options_registered = False
_runtime_registered = False


def register_runtime_components() -> None:
    """Register proven model adapters and engines."""

    global _runtime_registered

    with _registration_lock:
        if _runtime_registered:
            return

        from modules_nocturne.regional.adapters.sd15 import sd15_adapter
        from modules_nocturne.regional.adapters.sdxl import sdxl_adapter
        from modules_nocturne.regional.attention_engine import attention_decomposition_engine
        from modules_nocturne.regional.capabilities import capability_service

        for adapter in (sd15_adapter, sdxl_adapter):
            registered = capability_service.adapters.get(adapter.adapter_id)
            if registered is None:
                capability_service.adapters.register(adapter)
            elif registered is not adapter:
                raise RuntimeError(f"Conflicting Regional adapter registration: {adapter.adapter_id}")
        registered_engine = capability_service.engines.get(
            attention_decomposition_engine.engine_id
        )
        if registered_engine is None:
            capability_service.engines.register(attention_decomposition_engine)
        elif registered_engine is not attention_decomposition_engine:
            raise RuntimeError(
                "Conflicting Regional engine registration: "
                f"{attention_decomposition_engine.engine_id}"
            )
        _runtime_registered = True
        LOGGER.debug("Nocturne runtime components registered")


def register_options(
    options_templates: dict,
    *,
    options_section,
    option_info,
    option_html,
    categories,
    gradio,
) -> None:
    """Register Nocturne settings without loading a model or touching the GPU."""

    global _options_registered

    register_runtime_components()

    with _registration_lock:
        if _options_registered:
            return

        categories.register_category("nocturne", "Nocturne")

        options_templates.update(
            options_section(
                ("nocturne-regional", "Regional", "nocturne"),
                {
                    "nocturne_regional_intro": option_html("Defaults used by the native Regional workspace."),
                    "nocturne_regional_max_regions": option_info(
                        16,
                        "Maximum enabled regions",
                        gradio.Slider,
                        {"minimum": 1, "maximum": 64, "step": 1},
                    ),
                    "nocturne_regional_attention_memory_mb": option_info(
                        64,
                        "Regional attention working-memory budget (MiB)",
                        gradio.Slider,
                        {"minimum": 16, "maximum": 2048, "step": 16},
                    ).info(
                        "Limits how many region branches are evaluated together; lower values reduce peak VRAM use"
                    ),
                    "nocturne_regional_engine": option_info(
                        "Auto",
                        "Regional engine",
                        gradio.Dropdown,
                        {"choices": ("Auto",)},
                    ).info("Additional choices appear only when their runtime support is available"),
                },
            )
        )

        options_templates.update(
            options_section(
                ("nocturne-diagnostics", "Diagnostics / Experimental", "nocturne"),
                {
                    "nocturne_diagnostics_intro": option_html("Troubleshooting and experimental controls are disabled by default."),
                    "nocturne_diagnostics_attention_maps": option_info(False, "Capture attention maps"),
                    "nocturne_diagnostics_mask_dumps": option_info(False, "Write compiled mask diagnostics"),
                    "nocturne_diagnostics_detailed_timing": option_info(False, "Collect detailed timing"),
                    "nocturne_diagnostics_verbose_logging": option_info(False, "Enable verbose Nocturne logging"),
                    "nocturne_experimental_engines": option_info(False, "Enable experimental engines"),
                },
            )
        )

        _options_registered = True
        LOGGER.debug("Nocturne settings registered")

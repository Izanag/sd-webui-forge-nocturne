"""Native Nocturne UI surfaces."""

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from uuid import UUID

import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers, shared, ui_toprow
from modules_nocturne.regional.editor import (
    add_region,
    apply_geometry_edit,
    delete_region,
    duplicate_region,
    geometry_edit_between,
    initial_editor_plan,
    load_valid_editor_plan,
    move_region,
    plan_from_txt2img_settings,
    region_choices,
    render_layout_svg,
    render_mask_preview,
    raster_geometry_image,
    selected_region,
    update_canvas,
    update_engine_request,
    update_generation_options,
    update_global_prompts,
    update_polygon_geometry,
    update_raster_geometry,
    update_rectangle_geometry,
    update_region,
    validation_markdown,
)
from modules_nocturne.regional.capabilities import capability_service
from modules_nocturne.regional.errors import PlanError, PlanValidationError
from modules_nocturne.regional.model import PolygonGeometry, RasterMaskGeometry, RectGeometry, SeedMode
from modules_nocturne.regional.serialization import canonical_json, plan_hash
from modules_nocturne.regional.project import load_project, restore_metadata_file, save_project
from modules_nocturne.regional.validation import validate_plan


def _selected_updates(plan, selected_id, report=None):
    region = selected_region(plan, selected_id)
    interactive = region is not None
    values = (
        region.name if region else "",
        region.enabled if region else False,
        region.locked if region else False,
        region.hidden if region else False,
        region.positive if region else "",
        region.negative if region else "",
        region.inherit_global_positive if region else True,
        region.inherit_global_negative if region else True,
        region.weight if region else 1.0,
        region.priority if region else 0,
        region.feather_px if region else 0.0,
        region.grow_shrink_px if region else 0.0,
        region.guidance.start if region else 0.0,
        region.guidance.end if region else 1.0,
        region.seed.mode.value if region else SeedMode.DERIVED.value,
        region.seed.offset if region else 0,
    )
    updates = [gr.update(value=value, interactive=interactive) for value in values]
    geometry = region.geometry if region else None
    geometry_interactive = interactive and not region.locked
    is_rect = isinstance(geometry, RectGeometry)
    is_polygon = isinstance(geometry, PolygonGeometry)
    is_raster = isinstance(geometry, RasterMaskGeometry)
    geometry_name = (
        "Rectangle" if is_rect else "Polygon" if is_polygon else "Painted mask" if is_raster else "No geometry selected"
    )
    rect_values = (
        geometry.x if is_rect else 0.0,
        geometry.y if is_rect else 0.0,
        geometry.width if is_rect else 0.25,
        geometry.height if is_rect else 0.25,
    )
    polygon_value = (
        json.dumps([{"x": point.x, "y": point.y} for point in geometry.points], indent=2)
        if is_polygon
        else "[]"
    )
    report = report or validate_plan(plan)
    region_index = next((index for index, item in enumerate(plan.regions) if item.id == region.id), None) if region else None
    region_path = f"$.regions[{region_index}]" if region_index is not None else None
    relevant_issues = [issue for issue in report.issues if region_path and issue.path.startswith(region_path)]
    field_status = validation_markdown(type(report)(tuple(relevant_issues))) if relevant_issues else ""
    geometry_updates = (
        gr.update(value=f"**Geometry:** {geometry_name}"),
        *(gr.update(value=value, visible=is_rect, interactive=geometry_interactive and is_rect) for value in rect_values),
        gr.update(value=polygon_value, visible=is_polygon, interactive=geometry_interactive and is_polygon),
        gr.update(value=raster_geometry_image(region), visible=is_raster, interactive=geometry_interactive and is_raster),
        gr.update(visible=is_polygon, interactive=geometry_interactive and is_polygon),
        gr.update(visible=is_raster, interactive=geometry_interactive and is_raster),
        gr.update(value=field_status, visible=bool(field_status)),
    )
    uuid_text = f"`{region.id}`" if region else "No region selected."
    return (*updates, *geometry_updates, gr.update(value=uuid_text))


def _operation_updates(selected_id):
    interactive = selected_id is not None
    return tuple(gr.update(interactive=interactive) for _ in range(4))


def _snapshot(plan, selected_id, *, status=None, raw_value=None):
    report = validate_plan(plan)
    if selected_id:
        selected = UUID(str(selected_id))
        if selected not in {region.id for region in plan.regions}:
            selected = None
    else:
        selected = None
    if selected is None and plan.regions:
        selected = plan.regions[0].id
    serialized = canonical_json(plan)
    selected_value = str(selected) if selected else None
    common = (
        serialized,
        serialized,
        selected_value,
        status or validation_markdown(report),
        f"`{plan_hash(plan)}`",
        raw_value if raw_value is not None else serialized,
        render_layout_svg(plan, selected),
        gr.update(choices=region_choices(plan), value=selected_value),
    )
    options = plan.engine.options
    plan_controls = (
        gr.update(value=plan.global_prompt.positive),
        gr.update(value=plan.global_prompt.negative),
        gr.update(value=plan.canvas.width),
        gr.update(value=plan.canvas.height),
        gr.update(value=options.get("sampler", "Euler a")),
        gr.update(value=options.get("scheduler", "Automatic")),
        gr.update(value=options.get("steps", 32)),
        gr.update(value=options.get("cfg_scale", 6.0)),
        gr.update(value=options.get("batch_count", 1)),
        gr.update(value=options.get("batch_size", 1)),
        gr.update(value=options.get("seed", -1)),
        _engine_component_update(plan.engine.requested),
    )
    return (*common, *plan_controls, *_selected_updates(plan, selected, report), *_operation_updates(selected))


def _error_snapshot(last_valid_json, selected_id, error, *, raw_value=None):
    plan = load_valid_editor_plan(last_valid_json)
    if isinstance(error, PlanValidationError):
        status = validation_markdown(error.report)
    elif isinstance(error, PlanError):
        issue = error.issue
        status = f"- ✕ `{issue.code}` at `{issue.path}` — {issue.message}"
    else:
        status = f"- ✕ `editor.input.invalid` — {error}"
    return _snapshot(plan, selected_id, status=status, raw_value=raw_value)


def _apply_raw_plan(raw_value, last_valid_json, selected_id):
    try:
        plan = load_valid_editor_plan(raw_value)
        return _snapshot(plan, selected_id)
    except (PlanError, TypeError, ValueError) as error:
        return _error_snapshot(last_valid_json, selected_id, error, raw_value=raw_value)


def _mutate(last_valid_json, selected_id, mutation):
    try:
        plan = load_valid_editor_plan(last_valid_json)
        candidate, selected = mutation(plan, UUID(str(selected_id)) if selected_id else None)
        report = validate_plan(candidate)
        if not report.valid:
            raise PlanValidationError(report)
        return _snapshot(candidate, selected)
    except (PlanError, TypeError, ValueError) as error:
        return _error_snapshot(last_valid_json, selected_id, error)


def _geometry_history_result(snapshot, history, future):
    return (
        *snapshot,
        list(history),
        list(future),
        gr.update(interactive=bool(history)),
        gr.update(interactive=bool(future)),
    )


def _mutate_geometry(last_valid_json, selected_id, history, future, mutation):
    history = tuple(history or ())
    future = tuple(future or ())
    try:
        plan = load_valid_editor_plan(last_valid_json)
        selected = UUID(str(selected_id)) if selected_id else None
        if selected is None:
            return _raise_no_selection()
        candidate = mutation(plan, selected)
        report = validate_plan(candidate)
        if not report.valid:
            raise PlanValidationError(report)
        edit = geometry_edit_between(plan, candidate, selected)
        if edit.before == edit.after:
            return _geometry_history_result(_snapshot(candidate, selected), history, future)
        new_history = (*history, edit)[-100:]
        return _geometry_history_result(_snapshot(candidate, selected), new_history, ())
    except (PlanError, TypeError, ValueError) as error:
        return _geometry_history_result(_error_snapshot(last_valid_json, selected_id, error), history, future)


def _undo_geometry(last_valid_json, selected_id, history, future, undo):
    history = tuple(history or ())
    future = tuple(future or ())
    source = history if undo else future
    try:
        if not source:
            raise PlanError("editor.geometry.history_empty", "$.regions", "No geometry edit is available")
        plan = load_valid_editor_plan(last_valid_json)
        edit = source[-1]
        candidate = apply_geometry_edit(plan, edit, undo=undo)
        report = validate_plan(candidate)
        if not report.valid:
            raise PlanValidationError(report)
        if undo:
            history = history[:-1]
            future = (*future, edit)[-100:]
        else:
            future = future[:-1]
            history = (*history, edit)[-100:]
        return _geometry_history_result(_snapshot(candidate, edit.region_id), history, future)
    except (PlanError, TypeError, ValueError) as error:
        return _geometry_history_result(_error_snapshot(last_valid_json, selected_id, error), history, future)


def _select_region(plan_json, selected_id):
    plan = load_valid_editor_plan(plan_json)
    report = validate_plan(plan)
    region = selected_region(plan, selected_id)
    selected = str(region.id) if region else None
    return (
        selected,
        render_layout_svg(plan, selected),
        *_selected_updates(plan, selected, report),
        *_operation_updates(selected),
    )


def _preview_masks(plan_json):
    try:
        plan = load_valid_editor_plan(plan_json)
        image, compiled = render_mask_preview(plan)
        diagnostics = compiled.diagnostics
        text = (
            f"Coverage: **{diagnostics.covered_fraction:.2%}** · "
            f"Uncovered: **{diagnostics.uncovered_fraction:.2%}** · "
            f"Overlap: **{diagnostics.overlap_fraction:.2%}** · "
            f"Maximum overlap: **{diagnostics.maximum_overlap}**"
        )
        return image, text
    except (PlanError, TypeError, ValueError) as error:
        code = error.code if isinstance(error, PlanError) else "mask.preview.failed"
        return None, f"Preview unavailable: `{code}` — {error}"


def _load_project_file(project_path, last_valid_json, selected_id):
    try:
        if not project_path:
            raise PlanError("project.input.missing", "$", "Choose a project file first")
        plan = load_project(project_path)
        report = validate_plan(plan)
        if not report.valid:
            raise PlanValidationError(report)
        return _snapshot(plan, selected_id)
    except (OSError, PlanError, TypeError, ValueError) as error:
        return _error_snapshot(last_valid_json, selected_id, error)


def _restore_metadata_source(source_path, last_valid_json, selected_id):
    try:
        restored = restore_metadata_file(source_path)
        report = validate_plan(restored.metadata.plan)
        if not report.valid:
            raise PlanValidationError(report)
        status = (
            f"✓ Restored the exact canonical plan from {restored.source_kind}. "
            f"Plan hash: `{plan_hash(restored.metadata.plan)}`"
        )
        return _snapshot(restored.metadata.plan, selected_id, status=status)
    except (OSError, PlanError, TypeError, ValueError) as error:
        return _error_snapshot(last_valid_json, selected_id, error)


def _save_project_file(plan_json):
    plan = load_valid_editor_plan(plan_json)
    directory = Path(tempfile.mkdtemp(prefix="nocturne-project-"))
    destination = directory / "regional.nocturne-region.json"
    save_project(destination, plan)
    return str(destination)


def transfer_txt2img_plan(
    positive,
    negative,
    width,
    height,
    sampler_name,
    scheduler_name,
    step_count,
    cfg,
    count,
    size,
    base_seed,
):
    plan = plan_from_txt2img_settings(
        positive=positive or "",
        negative=negative or "",
        width=int(width),
        height=int(height),
        sampler=sampler_name,
        scheduler=scheduler_name,
        steps=int(step_count),
        cfg_scale=float(cfg),
        batch_count=int(count),
        batch_size=int(size),
        seed=int(base_seed),
    )
    return _snapshot(
        plan,
        None,
        status="✓ Imported txt2img prompt and generation settings into a new empty Regional plan.",
    )


def _capability_values():
    report = capability_service.report(getattr(shared, "sd_model", None))
    choices = ["auto", *(engine.engine_id for engine in report.eligible_engines)]
    if report.status == "supported":
        details = f"Adapter: `{report.adapter_id}`. Eligible engines: {', '.join(choices[1:]) or 'none'}."
        if report.expected_fallbacks:
            details += " Expected fallbacks: " + "; ".join(report.expected_fallbacks)
        return choices, len(choices) > 1, details
    return ["auto"], False, (
        f"Auto unavailable: `{report.reason_code or 'model.unsupported'}` — "
        f"{report.reason or 'Unsupported model.'}"
    )


def _capability_controls(current="auto"):
    choices, interactive, details = _capability_values()
    if current not in choices:
        choices = [*choices, current]
        details += f" Requested engine `{current}` is not currently eligible; the plan was not changed."
    cost_text, required_fallbacks = _engine_preflight_values(current)
    return (
        gr.update(choices=choices, value=current, interactive=interactive),
        details,
        gr.update(value=cost_text, visible=bool(cost_text)),
        gr.update(
            value=False,
            visible=bool(required_fallbacks),
            interactive=bool(required_fallbacks),
            info="Required before generation can accept: " + "; ".join(required_fallbacks)
            if required_fallbacks
            else None,
        ),
        (),
    )


def _engine_component_update(current):
    choices, interactive, _ = _capability_values()
    if current not in choices:
        choices = [*choices, current]
    return gr.update(choices=choices, value=current, interactive=interactive)


def _engine_preflight_values(current):
    report = capability_service.report(getattr(shared, "sd_model", None))
    if report.status != "supported":
        return "", ()
    engines = (
        report.eligible_engines
        if current == "auto"
        else tuple(engine for engine in report.eligible_engines if engine.engine_id == current)
    )
    cost_warnings = tuple(
        dict.fromkeys(engine.cost_warning for engine in engines if engine.cost_warning)
    )
    required_fallbacks = tuple(
        dict.fromkeys(
            (
                *report.expected_fallbacks,
                *(fallback for engine in engines for fallback in engine.expected_fallbacks),
            )
        )
    )
    cost_text = ""
    if cost_warnings:
        cost_text = "⚠ **Engine cost warning:** " + " ".join(cost_warnings)
    return cost_text, required_fallbacks


def _engine_preflight_controls(current):
    cost_text, required_fallbacks = _engine_preflight_values(current)
    return (
        gr.update(value=cost_text, visible=bool(cost_text)),
        gr.update(
            value=False,
            visible=bool(required_fallbacks),
            interactive=bool(required_fallbacks),
            info="Required before generation can accept: " + "; ".join(required_fallbacks)
            if required_fallbacks
            else None,
        ),
        (),
    )


def _accepted_fallbacks(current, acknowledged):
    _, required_fallbacks = _engine_preflight_values(current)
    return required_fallbacks if acknowledged else ()


def create_regional_interface(create_output_panel: Callable, *, head: str | None = None) -> gr.Blocks:
    """Create the canonical-plan Regional authoring workspace."""

    initial_plan = initial_editor_plan()
    initial_json = canonical_json(initial_plan)
    initial_engine_choices, initial_engine_interactive, initial_capability_status = _capability_values()
    initial_cost_warning, initial_required_fallbacks = _engine_preflight_values("auto")

    with gr.Blocks(analytics_enabled=False, head=head) as regional_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, id_part="regional")
        toprow.submit.value = "Generation unavailable"
        toprow.submit.interactive = False

        with gr.Column(elem_id="regional_workspace"):
            gr.HTML(
                """
                <style>
                    #regional_canvas { overflow:auto; max-height:42rem; }
                    #regional_canvas svg { display:block; width:100%; min-height:20rem; touch-action:none; }
                    #regional_canvas .nocturne-layout-preview {
                        border-radius:var(--radius-lg);
                        overflow:hidden;
                        width:var(--nocturne-canvas-zoom, 100%);
                        min-width:16rem;
                    }
                    #regional_canvas .nocturne-region-shape[data-region-id] { cursor:move; }
                    #regional_canvas .nocturne-geometry-handle { cursor:crosshair; }
                    #regional_canvas .nocturne-region-labels { pointer-events:none; user-select:none; }
                    #regional_geometry_pointer_bridge { display:none !important; }
                    #regional_workspace_status p { margin:.25rem 0; }
                    @media (max-width: 900px) { #regional_canvas svg { min-height:15rem; } }
                </style>
                <section class="nocturne-regional-intro" aria-labelledby="regional_workspace_title">
                    <h2 id="regional_workspace_title">Regional</h2>
                    <p>Author prompts and spatial regions in one restorable plan.</p>
                    <p role="status"><strong>Generation remains unavailable.</strong> Validation and mask preview are active.</p>
                </section>
                """,
                elem_id="regional_workspace_status",
            )

            if toprow.is_compact:
                toprow.create_inline_toprow_prompts()

            plan_bridge = gr.Textbox(value=initial_json, visible=False, elem_id="regional_plan_bridge")
            last_valid_plan = gr.State(initial_json)
            selected_region_id = gr.State(None)
            geometry_history = gr.State([])
            geometry_future = gr.State([])
            accepted_fallbacks = gr.State(())
            geometry_pointer_bridge = gr.Textbox(
                value="",
                container=False,
                elem_id="regional_geometry_pointer_bridge",
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=280):
                    layout_mode = gr.Dropdown(
                        choices=["Grid / Splits", "Rectangle", "Polygon", "Paint Mask"],
                        value="Rectangle",
                        label="New region tool",
                        elem_id="regional_layout_mode",
                    )
                    region_list = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Selected region",
                        info="Enabled regions use a filled marker.",
                        elem_id="regional_region_list",
                    )
                    with gr.Row():
                        add_button = gr.Button("Add", variant="primary", tooltip="Add a region with a new stable ID")
                        duplicate_button = gr.Button(
                            "Duplicate",
                            interactive=False,
                            tooltip="Duplicate the selected region with a new stable ID",
                        )
                        delete_button = gr.Button(
                            "Delete",
                            variant="stop",
                            interactive=False,
                            tooltip="Delete the selected region",
                        )
                    with gr.Row():
                        move_up_button = gr.Button(
                            "Move up",
                            interactive=False,
                            tooltip="Move selected region earlier",
                        )
                        move_down_button = gr.Button(
                            "Move down",
                            interactive=False,
                            tooltip="Move selected region later",
                        )

                    with gr.Accordion("Selected region", open=True):
                        region_name = gr.Textbox(label="Name", interactive=False)
                        with gr.Row():
                            region_enabled = gr.Checkbox(label="Enabled", interactive=False)
                            region_locked = gr.Checkbox(label="Locked", interactive=False)
                            region_hidden = gr.Checkbox(label="Hidden", interactive=False)
                        local_positive = gr.Textbox(label="Local prompt", lines=3, interactive=False)
                        local_negative = gr.Textbox(label="Local negative prompt", lines=3, interactive=False)
                        with gr.Row():
                            inherit_positive = gr.Checkbox(label="Inherit global prompt", value=True, interactive=False)
                            inherit_negative = gr.Checkbox(label="Inherit global negative", value=True, interactive=False)
                        with gr.Row():
                            region_weight = gr.Slider(0.01, 8.0, value=1.0, step=0.01, label="Weight", interactive=False)
                            region_priority = gr.Number(value=0, precision=0, label="Priority", interactive=False)
                        with gr.Row():
                            region_feather = gr.Slider(0, 256, value=0, step=1, label="Feather (px)", interactive=False)
                            region_grow = gr.Slider(-128, 128, value=0, step=1, label="Grow / shrink (px)", interactive=False)
                        with gr.Row():
                            guidance_start = gr.Slider(0, 1, value=0, step=0.01, label="Guidance start", interactive=False)
                            guidance_end = gr.Slider(0, 1, value=1, step=0.01, label="Guidance end", interactive=False)
                        with gr.Row():
                            seed_mode = gr.Dropdown(
                                choices=[mode.value for mode in SeedMode],
                                value=SeedMode.DERIVED.value,
                                label="Region seed mode",
                                interactive=False,
                            )
                            seed_offset = gr.Number(value=0, precision=0, label="Seed offset", interactive=False)
                        with gr.Accordion("Geometry", open=True):
                            with gr.Row():
                                geometry_kind = gr.Markdown("**Geometry:** No geometry selected")
                                undo_geometry_button = gr.Button(
                                    "Undo geometry",
                                    interactive=False,
                                    tooltip="Undo the most recent committed geometry edit",
                                )
                                redo_geometry_button = gr.Button(
                                    "Redo geometry",
                                    interactive=False,
                                    tooltip="Redo the most recently undone geometry edit",
                                )
                            with gr.Row():
                                rect_x = gr.Slider(0, 1, value=0, step=0.001, label="X", visible=False)
                                rect_y = gr.Slider(0, 1, value=0, step=0.001, label="Y", visible=False)
                            with gr.Row():
                                rect_width = gr.Slider(0.001, 1, value=0.25, step=0.001, label="Width", visible=False)
                                rect_height = gr.Slider(0.001, 1, value=0.25, step=0.001, label="Height", visible=False)
                            polygon_points = gr.Code(
                                value="[]",
                                language="json",
                                label="Normalised polygon points",
                                lines=10,
                                visible=False,
                            )
                            apply_polygon_button = gr.Button("Apply polygon points", visible=False)
                            raster_editor = gr.ImageEditor(
                                label="Painted mask",
                                type="pil",
                                image_mode="L",
                                format="png",
                                sources=["upload", "clipboard"],
                                brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", color_mode="fixed"),
                                eraser=gr.Eraser(),
                                layers=False,
                                canvas_size=(512, 512),
                                visible=False,
                            )
                            apply_raster_button = gr.Button("Apply painted mask", visible=False)
                            region_field_validation = gr.Markdown("", visible=False)
                        gr.Markdown(
                            "LoRA and extra-network tags affect the whole generation, including tags written in a local prompt."
                        )

                with gr.Column(scale=5, min_width=420):
                    canvas_preview = gr.HTML(
                        value=render_layout_svg(initial_plan),
                        elem_id="regional_canvas",
                    )
                    canvas_zoom = gr.Slider(
                        25,
                        300,
                        value=100,
                        step=5,
                        label="Canvas zoom (%)",
                        info="Scroll the canvas to pan. Zoom and pan do not change normalised plan coordinates.",
                    )
                    with gr.Row():
                        canvas_width = gr.Slider(64, 2048, value=1024, step=8, label="Canvas width")
                        canvas_height = gr.Slider(64, 2048, value=1024, step=8, label="Canvas height")
                    with gr.Accordion("Generation controls", open=True):
                        with gr.Row():
                            engine_choice = gr.Dropdown(
                                choices=initial_engine_choices,
                                value="auto",
                                label="Regional engine",
                                interactive=initial_engine_interactive,
                            )
                            refresh_capabilities = gr.Button(
                                "Refresh support",
                                tooltip="Re-check the loaded model and registered Regional engines",
                            )
                        capability_status = gr.Markdown(
                            initial_capability_status,
                            elem_id="regional_capability_status",
                        )
                        engine_cost_warning = gr.Markdown(
                            initial_cost_warning,
                            visible=bool(initial_cost_warning),
                            elem_id="regional_engine_cost_warning",
                        )
                        fallback_acknowledgement = gr.Checkbox(
                            label="Accept required engine fallbacks",
                            value=False,
                            visible=bool(initial_required_fallbacks),
                            interactive=bool(initial_required_fallbacks),
                            info="Required before generation can accept: " + "; ".join(initial_required_fallbacks)
                            if initial_required_fallbacks
                            else None,
                            elem_id="regional_fallback_acknowledgement",
                        )
                        with gr.Row():
                            sampler = gr.Dropdown(
                                choices=sd_samplers.visible_sampler_names(),
                                value="Euler a",
                                label="Sampling method",
                            )
                            scheduler = gr.Dropdown(
                                choices=[item.label for item in sd_schedulers.schedulers],
                                value="Automatic",
                                label="Schedule type",
                            )
                        with gr.Row():
                            steps = gr.Slider(1, 150, value=32, step=1, label="Sampling steps")
                            cfg_scale = gr.Slider(1, 24, value=6, step=0.5, label="CFG scale")
                        with gr.Row():
                            batch_count = gr.Slider(1, 128, value=1, step=1, label="Batch count")
                            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch size")
                            seed = gr.Number(value=-1, precision=0, label="Seed")
                        gr.Checkbox(
                            label="Hires / refiner",
                            value=False,
                            interactive=False,
                            info="Unavailable until the active adapter proves pass support.",
                        )
                    with gr.Row():
                        validate_button = gr.Button("Validate", variant="secondary")
                        preview_button = gr.Button("Preview masks", variant="secondary")
                    validation_status = gr.Markdown("✓ Plan is structurally valid.", elem_id="regional_validation_status")
                    plan_hash_display = gr.Markdown(f"`{plan_hash(initial_plan)}`", label="Plan hash")
                    with gr.Row():
                        mask_preview = gr.Image(
                            label="Compiled mask preview",
                            type="pil",
                            interactive=False,
                            height=320,
                        )
                        mask_diagnostics = gr.Markdown(
                            "Run preview to inspect coverage.",
                            elem_id="regional_mask_summary",
                        )

                    with gr.Accordion("Advanced plan details", open=False):
                        region_uuid = gr.Markdown("No region selected.", label="Stable region UUID")
                        with gr.Row():
                            project_upload = gr.File(
                                label="Open project",
                                file_types=[".json"],
                                type="filepath",
                                file_count="single",
                            )
                            project_download = gr.File(
                                label="Saved project",
                                interactive=False,
                            )
                            metadata_upload = gr.File(
                                label="Restore from PNG or sidecar",
                                file_types=[".png", ".json"],
                                type="filepath",
                                file_count="single",
                            )
                        with gr.Row():
                            load_project_button = gr.Button("Open project")
                            save_project_button = gr.Button("Save project")
                            restore_metadata_button = gr.Button("Restore metadata")
                        raw_plan = gr.Code(
                            value=initial_json,
                            language="json",
                            label="Canonical plan",
                            lines=18,
                        )
                        apply_raw_button = gr.Button("Apply canonical plan")

            with gr.Accordion("Scripts", open=False, elem_id="regional_script_container"):
                scripts.scripts_regional.prepare_ui()
                scripts.scripts_regional.setup_ui(elem_id="regional_script_list")

            create_output_panel("regional", shared.opts.outdir_txt2img_samples, toprow)

        selected_editor_components = [
            region_name,
            region_enabled,
            region_locked,
            region_hidden,
            local_positive,
            local_negative,
            inherit_positive,
            inherit_negative,
            region_weight,
            region_priority,
            region_feather,
            region_grow,
            guidance_start,
            guidance_end,
            seed_mode,
            seed_offset,
        ]
        geometry_components = [
            geometry_kind,
            rect_x,
            rect_y,
            rect_width,
            rect_height,
            polygon_points,
            raster_editor,
            apply_polygon_button,
            apply_raster_button,
            region_field_validation,
            region_uuid,
        ]
        selected_components = [*selected_editor_components, *geometry_components]
        common_outputs = [
            plan_bridge,
            last_valid_plan,
            selected_region_id,
            validation_status,
            plan_hash_display,
            raw_plan,
            canvas_preview,
            region_list,
        ]
        full_outputs = [
            *common_outputs,
            toprow.prompt,
            toprow.negative_prompt,
            canvas_width,
            canvas_height,
            sampler,
            scheduler,
            steps,
            cfg_scale,
            batch_count,
            batch_size,
            seed,
            engine_choice,
            *selected_components,
            duplicate_button,
            delete_button,
            move_up_button,
            move_down_button,
        ]
        prompt_outputs = [
            *common_outputs,
            *selected_components,
            duplicate_button,
            delete_button,
            move_up_button,
            move_down_button,
        ]
        geometry_history_outputs = [
            *full_outputs,
            geometry_history,
            geometry_future,
            undo_geometry_button,
            redo_geometry_button,
        ]

        def add_action(plan_json, selected_id, mode):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, _: add_region(plan, "Rectangle" if mode == "Grid / Splits" else mode),
            )

        def duplicate_action(plan_json, selected_id):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: duplicate_region(plan, selected)
                if selected
                else (_raise_no_selection()),
            )

        def delete_action(plan_json, selected_id):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: delete_region(plan, selected)
                if selected
                else (_raise_no_selection()),
            )

        def move_action(plan_json, selected_id, offset):
            def mutation(plan, selected):
                if selected is None:
                    return _raise_no_selection()
                return move_region(plan, selected, offset), selected

            return _mutate(plan_json, selected_id, mutation)

        def prompt_action(plan_json, selected_id, positive, negative):
            result = _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (update_global_prompts(plan, positive, negative), selected),
            )
            return (*result[:8], *result[20:])

        def canvas_action(plan_json, selected_id, width, height):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (update_canvas(plan, width, height), selected),
            )

        def generation_action(plan_json, selected_id, sampler_name, scheduler_name, step_count, cfg, count, size, base_seed):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (
                    update_generation_options(
                        plan,
                        sampler=sampler_name,
                        scheduler=scheduler_name,
                        steps=int(step_count),
                        cfg_scale=float(cfg),
                        batch_count=int(count),
                        batch_size=int(size),
                        seed=int(base_seed),
                    ),
                    selected,
                ),
            )

        def engine_action(plan_json, selected_id, requested):
            return _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (update_engine_request(plan, requested), selected),
            )

        def region_action(plan_json, selected_id, *values):
            def mutation(plan, selected):
                if selected is None:
                    return _raise_no_selection()
                (
                    name,
                    enabled,
                    locked,
                    hidden,
                    positive,
                    negative,
                    inherit_global_positive,
                    inherit_global_negative,
                    weight,
                    priority,
                    feather_px,
                    grow_shrink_px,
                    start,
                    end,
                    mode,
                    offset,
                ) = values
                current = selected_region(plan, selected)
                seed = replace(current.seed, mode=SeedMode(mode), offset=int(offset))
                guidance = replace(current.guidance, start=float(start), end=float(end))
                updated = update_region(
                    plan,
                    selected,
                    name=name,
                    enabled=bool(enabled),
                    locked=bool(locked),
                    hidden=bool(hidden),
                    positive=positive,
                    negative=negative,
                    inherit_global_positive=bool(inherit_global_positive),
                    inherit_global_negative=bool(inherit_global_negative),
                    weight=float(weight),
                    priority=int(priority),
                    feather_px=float(feather_px),
                    grow_shrink_px=float(grow_shrink_px),
                    guidance=guidance,
                    seed=seed,
                )
                return updated, selected

            return _mutate(plan_json, selected_id, mutation)

        def rectangle_action(plan_json, selected_id, history, future, x, y, width, height):
            return _mutate_geometry(
                plan_json,
                selected_id,
                history,
                future,
                lambda plan, selected: update_rectangle_geometry(
                    plan,
                    selected,
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                )
            )

        def polygon_action(plan_json, selected_id, history, future, points):
            return _mutate_geometry(
                plan_json,
                selected_id,
                history,
                future,
                lambda plan, selected: update_polygon_geometry(plan, selected, points),
            )

        def raster_action(plan_json, selected_id, history, future, editor_value):
            def mutation(plan, selected):
                image = editor_value
                if isinstance(editor_value, dict):
                    image = editor_value.get("composite")
                    if image is None:
                        image = editor_value.get("background")
                return update_raster_geometry(plan, selected, image)

            return _mutate_geometry(plan_json, selected_id, history, future, mutation)

        def pointer_geometry_action(plan_json, selected_id, history, future, payload):
            try:
                if not isinstance(payload, str) or len(payload) > 64 * 1024:
                    raise PlanError("editor.geometry.pointer.invalid", "$.geometry", "Canvas geometry update is invalid")
                document = json.loads(payload)
                if not isinstance(document, dict) or document.get("region_id") != selected_id:
                    raise PlanError(
                        "editor.geometry.pointer.selection_mismatch",
                        "$.geometry",
                        "Canvas geometry update does not match the selected region",
                    )
                geometry_type = document.get("type")
                if geometry_type == "rect":
                    mutation = lambda plan, selected: update_rectangle_geometry(
                        plan,
                        selected,
                        x=document.get("x"),
                        y=document.get("y"),
                        width=document.get("width"),
                        height=document.get("height"),
                    )
                elif geometry_type == "polygon":
                    mutation = lambda plan, selected: update_polygon_geometry(
                        plan,
                        selected,
                        document.get("points"),
                    )
                else:
                    raise PlanError("editor.geometry.pointer.unsupported", "$.geometry.type", "Unsupported canvas geometry update")
                return _mutate_geometry(plan_json, selected_id, history, future, mutation)
            except (json.JSONDecodeError, PlanError, TypeError, ValueError) as error:
                return _geometry_history_result(
                    _error_snapshot(plan_json, selected_id, error),
                    tuple(history or ()),
                    tuple(future or ()),
                )

        add_button.click(add_action, inputs=[last_valid_plan, selected_region_id, layout_mode], outputs=full_outputs)
        duplicate_button.click(duplicate_action, inputs=[last_valid_plan, selected_region_id], outputs=full_outputs)
        delete_button.click(delete_action, inputs=[last_valid_plan, selected_region_id], outputs=full_outputs)
        move_up_button.click(
            lambda plan, selected: move_action(plan, selected, -1),
            inputs=[last_valid_plan, selected_region_id],
            outputs=full_outputs,
        )
        move_down_button.click(
            lambda plan, selected: move_action(plan, selected, 1),
            inputs=[last_valid_plan, selected_region_id],
            outputs=full_outputs,
        )

        region_list.input(
            _select_region,
            inputs=[last_valid_plan, region_list],
            outputs=[
                selected_region_id,
                canvas_preview,
                *selected_components,
                duplicate_button,
                delete_button,
                move_up_button,
                move_down_button,
            ],
            show_progress=False,
        )
        canvas_zoom.input(
            fn=None,
            inputs=[canvas_zoom],
            outputs=[],
            js="""
                (zoom) => {
                    const root = typeof gradioApp === "function" ? gradioApp() : document;
                    const canvas = root.querySelector("#regional_canvas");
                    if (canvas) canvas.style.setProperty("--nocturne-canvas-zoom", `${zoom}%`);
                    return [];
                }
            """,
            show_progress=False,
        )
        toprow.prompt.change(
            prompt_action,
            inputs=[last_valid_plan, selected_region_id, toprow.prompt, toprow.negative_prompt],
            outputs=prompt_outputs,
            show_progress=False,
            trigger_mode="always_last",
        )
        toprow.negative_prompt.change(
            prompt_action,
            inputs=[last_valid_plan, selected_region_id, toprow.prompt, toprow.negative_prompt],
            outputs=prompt_outputs,
            show_progress=False,
            trigger_mode="always_last",
        )
        for dimension in (canvas_width, canvas_height):
            dimension.input(
                canvas_action,
                inputs=[last_valid_plan, selected_region_id, canvas_width, canvas_height],
                outputs=full_outputs,
                show_progress=False,
                trigger_mode="always_last",
            )

        generation_inputs = [sampler, scheduler, steps, cfg_scale, batch_count, batch_size, seed]
        for component in generation_inputs:
            component.input(
                generation_action,
                inputs=[last_valid_plan, selected_region_id, *generation_inputs],
                outputs=full_outputs,
                show_progress=False,
                trigger_mode="always_last",
            )
        engine_choice.input(
            engine_action,
            inputs=[last_valid_plan, selected_region_id, engine_choice],
            outputs=full_outputs,
            show_progress=False,
        ).then(
            _engine_preflight_controls,
            inputs=[engine_choice],
            outputs=[engine_cost_warning, fallback_acknowledgement, accepted_fallbacks],
            show_progress=False,
        )
        refresh_capabilities.click(
            _capability_controls,
            inputs=[engine_choice],
            outputs=[
                engine_choice,
                capability_status,
                engine_cost_warning,
                fallback_acknowledgement,
                accepted_fallbacks,
            ],
            show_progress=False,
        )
        fallback_acknowledgement.input(
            _accepted_fallbacks,
            inputs=[engine_choice, fallback_acknowledgement],
            outputs=[accepted_fallbacks],
            show_progress=False,
        )

        region_inputs = selected_editor_components
        for component in region_inputs:
            component.input(
                region_action,
                inputs=[last_valid_plan, selected_region_id, *region_inputs],
                outputs=full_outputs,
                show_progress=False,
                trigger_mode="always_last",
            )

        rectangle_inputs = [rect_x, rect_y, rect_width, rect_height]
        for component in rectangle_inputs:
            component.input(
                rectangle_action,
                inputs=[
                    last_valid_plan,
                    selected_region_id,
                    geometry_history,
                    geometry_future,
                    *rectangle_inputs,
                ],
                outputs=geometry_history_outputs,
                show_progress=False,
                trigger_mode="always_last",
            )
        apply_polygon_button.click(
            polygon_action,
            inputs=[last_valid_plan, selected_region_id, geometry_history, geometry_future, polygon_points],
            outputs=geometry_history_outputs,
            show_progress=False,
        )
        apply_raster_button.click(
            raster_action,
            inputs=[last_valid_plan, selected_region_id, geometry_history, geometry_future, raster_editor],
            outputs=geometry_history_outputs,
            show_progress=False,
        )
        geometry_pointer_bridge.input(
            pointer_geometry_action,
            inputs=[
                last_valid_plan,
                selected_region_id,
                geometry_history,
                geometry_future,
                geometry_pointer_bridge,
            ],
            outputs=geometry_history_outputs,
            show_progress=False,
            trigger_mode="always_last",
        )
        undo_geometry_button.click(
            lambda plan, selected, history, future: _undo_geometry(
                plan,
                selected,
                history,
                future,
                True,
            ),
            inputs=[last_valid_plan, selected_region_id, geometry_history, geometry_future],
            outputs=geometry_history_outputs,
            show_progress=False,
        )
        redo_geometry_button.click(
            lambda plan, selected, history, future: _undo_geometry(
                plan,
                selected,
                history,
                future,
                False,
            ),
            inputs=[last_valid_plan, selected_region_id, geometry_history, geometry_future],
            outputs=geometry_history_outputs,
            show_progress=False,
        )

        validate_button.click(
            lambda value, selected: _snapshot(load_valid_editor_plan(value), selected),
            inputs=[last_valid_plan, selected_region_id],
            outputs=full_outputs,
            show_progress=False,
        )
        preview_button.click(
            _preview_masks,
            inputs=[last_valid_plan],
            outputs=[mask_preview, mask_diagnostics],
        )
        apply_raw_button.click(
            _apply_raw_plan,
            inputs=[raw_plan, last_valid_plan, selected_region_id],
            outputs=full_outputs,
        )
        load_project_button.click(
            _load_project_file,
            inputs=[project_upload, last_valid_plan, selected_region_id],
            outputs=full_outputs,
        )
        restore_metadata_button.click(
            _restore_metadata_source,
            inputs=[metadata_upload, last_valid_plan, selected_region_id],
            outputs=full_outputs,
        )
        save_project_button.click(
            _save_project_file,
            inputs=[last_valid_plan],
            outputs=[project_download],
        )

    regional_interface.nocturne_transfer_outputs = full_outputs
    return regional_interface


def _raise_no_selection():
    raise PlanError("editor.region.selection_required", "$.regions", "Select a region first")

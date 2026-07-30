"""Native Nocturne UI surfaces."""

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from uuid import UUID

import gradio as gr

from modules import scripts, sd_models, sd_samplers, sd_schedulers, shared, ui_toprow
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
    render_region_table,
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
    update_refiner_policy,
    validation_markdown,
)
from modules_nocturne.regional.capabilities import capability_service
from modules_nocturne.regional.errors import PlanError, PlanValidationError
from modules_nocturne.regional.generation import required_plan_fallbacks
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
        render_region_table(plan, selected),
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
        gr.update(value=options.get("subseed", -1)),
        gr.update(value=options.get("subseed_strength", 0.0)),
        gr.update(value=options.get("seed_resize_from_width", 0)),
        gr.update(value=options.get("seed_resize_from_height", 0)),
        _engine_component_update(plan.engine.requested),
        gr.update(value=options.get("hires_enabled", False)),
        gr.update(value=options.get("hires_upscaler", "Latent")),
        gr.update(value=options.get("hires_steps", 0)),
        gr.update(value=options.get("hires_denoising_strength", 0.6)),
        gr.update(value=options.get("hires_scale", 2.0)),
        gr.update(value=options.get("hires_width", 0)),
        gr.update(value=options.get("hires_height", 0)),
        gr.update(value=options.get("hires_distilled_cfg_scale", 3.0)),
        gr.update(value=options.get("hires_cfg_scale", options.get("cfg_scale", 6.0))),
        gr.update(value=options.get("refiner_enabled", False)),
        gr.update(value=options.get("refiner_checkpoint", "")),
        gr.update(value=options.get("refiner_switch_mode", "steps")),
        gr.update(value=options.get("refiner_switch_at", 0.8)),
        gr.update(value=options.get("refiner_cfg_scale", 0.0)),
        gr.update(value=plan.passes.refiner),
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
        gr.update(value=selected),
        render_layout_svg(plan, selected),
        render_region_table(plan, selected),
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


def _loaded_capability_report():
    from modules.sd_models import FakeInitialModel

    model = getattr(shared, "sd_model", None)
    if model is None or isinstance(model, FakeInitialModel):
        return None
    return capability_service.report(model)


def _capability_values():
    report = _loaded_capability_report()
    if report is None:
        return ["auto"], False, (
            "Model support is checked during generation after Forge loads the selected checkpoint."
        )
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


def _engine_component_update(current):
    choices, interactive, _ = _capability_values()
    if current not in choices:
        choices = [*choices, current]
    return gr.update(choices=choices, value=current, interactive=interactive)


def _engine_preflight_values(current, plan=None):
    report = _loaded_capability_report()
    plan_fallbacks = required_plan_fallbacks(plan) if plan is not None else ()
    if report is None:
        return "", plan_fallbacks
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
                *plan_fallbacks,
            )
        )
    )
    cost_text = ""
    if cost_warnings:
        cost_text = "⚠ **Engine cost warning:** " + " ".join(cost_warnings)
    return cost_text, required_fallbacks


def _engine_preflight_controls(current, plan_json):
    plan = load_valid_editor_plan(plan_json)
    cost_text, required_fallbacks = _engine_preflight_values(current, plan)
    report = _loaded_capability_report()
    generation_available = report is None or report.status == "supported"
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
        False,
        gr.update(
            value="Generate" if generation_available else "Generation unavailable",
            interactive=generation_available and not required_fallbacks,
        ),
    )


def _accepted_fallback_controls(current, acknowledged, plan_json):
    plan = load_valid_editor_plan(plan_json)
    _, required_fallbacks = _engine_preflight_values(current, plan)
    report = _loaded_capability_report()
    accepted = bool(acknowledged)
    ready = (report is None or report.status == "supported") and (
        not required_fallbacks or bool(acknowledged)
    )
    return accepted, gr.update(
        value="Generate" if report is None or report.status == "supported" else "Generation unavailable",
        interactive=ready,
    )


def _regional_generate_function(
    id_task,
    request: gr.Request,
    plan_json,
    accepted_fallbacks,
    *script_args,
):
    from contextlib import closing

    from modules import processing
    from modules.ui import plaintext_to_html
    from modules_nocturne.regional.processing import StableDiffusionProcessingRegional
    from modules_nocturne.regional.project import build_metadata, save_sidecar

    unverified_scripts = scripts.scripts_regional.active_unverified_script_titles(
        script_args
    )
    if unverified_scripts:
        gr.Warning(
            "Attempting unverified Regional scripts: "
            + ", ".join(unverified_scripts)
            + ". Disable the setting if generation becomes unstable."
        )

    plan = load_valid_editor_plan(plan_json)
    with closing(
        StableDiffusionProcessingRegional.from_plan(
            plan,
            sd_model=getattr(shared, "sd_model", None),
            outpath_samples=shared.opts.outdir_samples
            or shared.opts.outdir_txt2img_samples,
            outpath_grids=shared.opts.outdir_grids
            or shared.opts.outdir_txt2img_grids,
            scripts_runner=scripts.scripts_regional,
            script_args=script_args,
            accept_required_fallbacks=bool(accepted_fallbacks),
        )
    ) as regional:
        regional.user = request.username
        processed = scripts.scripts_regional.run(regional, *script_args)
        if processed is None:
            processed = processing.process_images(regional)
        processing.process_extra_images(processed)
        authorized = regional.authorized_plan
        if authorized is None:
            raise RuntimeError("Regional generation completed without authorization")
        metadata = build_metadata(
            authorized.plan,
            selected_engine=authorized.engine.engine_id,
            engine_version=authorized.engine.engine_version,
            engine_runtime_options=regional.regional_engine_runtime_options,
            adapter_id=authorized.adapter_id,
            accepted_fallbacks=authorized.accepted_fallbacks,
            resolved_seeds=tuple(regional.regional_resolved_seeds),
        )
        for image in processed.images + processed.extra_images:
            image.info.update(metadata.fields)
            saved_path = getattr(image, "already_saved_as", None)
            if metadata.sidecar_required and saved_path:
                save_sidecar(saved_path, metadata)

    generation_info = processed.js()
    if shared.opts.samples_log_stdout:
        print(generation_info)
    if shared.opts.do_not_show_images:
        processed.images = []

    if processed.video_path is None:
        gallery = gr.update(
            value=processed.images + processed.extra_images,
            visible=True,
        )
        player = gr.update(value=None, visible=False)
    else:
        gallery = gr.update(value=None, visible=False)
        player = gr.update(value=processed.video_path, visible=True)
    engine_choices, engine_interactive, capability_text = _capability_values()
    requested_engine = authorized.plan.engine.requested
    if requested_engine not in engine_choices:
        requested_engine = "auto"
    return (
        gallery,
        player,
        generation_info,
        plaintext_to_html(processed.info),
        gr.update(
            choices=engine_choices,
            value=requested_engine,
            interactive=engine_interactive,
        ),
        gr.update(value=capability_text),
        plaintext_to_html(processed.comments, classname="comments"),
    )


def _regional_generate(
    id_task: str,
    request: gr.Request,
    plan_json,
    accepted_fallbacks,
    *script_args,
):
    from modules_forge import main_thread

    return main_thread.run_and_wait_result(
        _regional_generate_function,
        id_task,
        request,
        plan_json,
        accepted_fallbacks,
        *script_args,
    )


def _refiner_checkpoint_choices() -> list[str]:
    return [
        "",
        *[
            checkpoint.name
            for checkpoint in sd_models.checkpoints_list.values()
            if checkpoint.metadata.get("modelspec.architecture")
            == "stable-diffusion-xl-v1-refiner"
        ],
    ]


def create_regional_interface(create_output_panel: Callable, *, head: str | None = None) -> gr.Blocks:
    """Create the canonical-plan Regional authoring workspace."""

    initial_plan = initial_editor_plan()
    initial_json = canonical_json(initial_plan)
    initial_engine_choices, initial_engine_interactive, initial_capability_status = _capability_values()
    initial_cost_warning, initial_required_fallbacks = _engine_preflight_values(
        "auto",
        initial_plan,
    )
    initial_capability_report = _loaded_capability_report()
    initial_generation_available = (
        (initial_capability_report is None or initial_capability_report.status == "supported")
        and not initial_required_fallbacks
    )

    with gr.Blocks(analytics_enabled=False, head=head) as regional_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, id_part="regional")
        toprow.submit.value = (
            "Generate"
            if initial_capability_report is None or initial_capability_report.status == "supported"
            else "Generation unavailable"
        )
        toprow.submit.interactive = initial_generation_available

        with gr.Column(elem_id="regional_workspace"):
            gr.HTML(
                """
                <style>
                    #tab_regional {
                        border:0 !important;
                        border-radius:0 !important;
                        padding-inline:0 !important;
                    }
                    #regional_canvas {
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        overflow:hidden !important;
                        padding:.75rem;
                        height:clamp(22rem, 42vh, 34rem);
                    }
                    #regional_canvas.nocturne-canvas-zoomed {
                        align-items:flex-start;
                        justify-content:flex-start;
                        overflow:auto !important;
                    }
                    #regional_canvas > div,
                    #regional_canvas > div > .prose {
                        width:100% !important;
                        height:100% !important;
                        max-width:none !important;
                    }
                    #regional_canvas > div > .prose {
                        display:flex !important;
                        align-items:center;
                        justify-content:center;
                    }
                    #regional_canvas svg {
                        display:block;
                        width:100%;
                        height:100%;
                        min-height:0;
                        touch-action:none;
                    }
                    #regional_canvas .nocturne-layout-preview {
                        flex:0 0 auto;
                        overflow:hidden;
                        width:min(64rem, 92%);
                        margin:auto;
                        max-width:none;
                    }
                    #regional_canvas .nocturne-region-shape[data-region-id] { cursor:move; }
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="n"],
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="s"] { cursor:ns-resize; }
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="e"],
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="w"] { cursor:ew-resize; }
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="nw"],
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="se"] { cursor:nwse-resize; }
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="ne"],
                    #regional_canvas .nocturne-geometry-handle[data-rect-corner="sw"] { cursor:nesw-resize; }
                    #regional_canvas .nocturne-geometry-handle[data-point-index] { cursor:crosshair; }
                    #regional_canvas .nocturne-region-labels { pointer-events:none; user-select:none; }
                    #regional_gallery,
                    #regional_gallery > div {
                        min-height:clamp(28rem, 48vh, 36rem);
                    }
                    #regional_canvas_toolbar {
                        align-items:center;
                        justify-content:flex-end;
                        min-height:2rem;
                    }
                    #regional_snap_to_grid {
                        flex:0 0 auto;
                        min-width:0;
                    }
                    #regional_geometry_pointer_bridge,
                    #regional_region_selection_bridge,
                    #regional_region_list { display:none !important; }
                    #regional_region_table .nocturne-region-table-wrap {
                        max-height:12rem;
                        overflow:auto;
                        border:1px solid var(--block-border-color);
                        border-radius:var(--block-radius);
                    }
                    #regional_region_table .nocturne-region-table {
                        width:100%;
                        border-collapse:collapse;
                    }
                    #regional_region_table td { padding:0; }
                    #regional_region_table button {
                        display:grid;
                        grid-template-columns:.8rem 2rem minmax(0, 1fr) auto;
                        align-items:center;
                        gap:.55rem;
                        width:100%;
                        padding:.48rem .6rem;
                        color:var(--body-text-color);
                        text-align:left;
                        background:transparent;
                        border:0;
                        border-bottom:1px solid var(--block-border-color);
                        cursor:pointer;
                    }
                    #regional_region_table tr:last-child button { border-bottom:0; }
                    #regional_region_table button:hover { background:var(--block-label-background-fill); }
                    #regional_region_table .is-selected button {
                        background:var(--button-secondary-background-fill);
                        box-shadow:inset .2rem 0 0 var(--region-color, transparent);
                    }
                    #regional_region_table .nocturne-region-swatch {
                        width:.72rem;
                        height:.72rem;
                        border-radius:50%;
                        background:var(--region-color);
                        box-shadow:0 0 0 1px color-mix(in srgb, var(--region-color) 70%, white);
                    }
                    #regional_region_table .nocturne-region-name {
                        overflow:hidden;
                        text-overflow:ellipsis;
                        white-space:nowrap;
                    }
                    #regional_region_table .nocturne-region-state {
                        color:var(--body-text-color-subdued);
                        font-size:.82em;
                    }
                    #regional_region_table .nocturne-region-table-empty {
                        padding:.65rem;
                        color:var(--body-text-color-subdued);
                        border:1px dashed var(--block-border-color);
                        border-radius:var(--block-radius);
                    }
                    #regional_workspace_status { display:none !important; }
                    @media (max-width: 900px) {
                        #regional_editor_layout { flex-direction:column; }
                        #regional_editor_layout > .gradio-column {
                            width:100%;
                            min-width:0 !important;
                        }
                        #regional_canvas { height:clamp(22rem, 55vh, 34rem); }
                    }
                </style>
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
            accepted_fallbacks = gr.State(False)
            generation_task = gr.Textbox(
                value="",
                visible=False,
                elem_id="regional_generation_task",
            )
            geometry_pointer_bridge = gr.Textbox(
                value="",
                container=False,
                elem_id="regional_geometry_pointer_bridge",
            )
            region_selection_bridge = gr.Textbox(
                value="",
                container=False,
                elem_id="regional_region_selection_bridge",
            )

            with gr.Row(equal_height=False, elem_id="regional_editor_layout"):
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
                        visible=False,
                        elem_id="regional_region_list",
                    )
                    region_table = gr.HTML(
                        value=render_region_table(initial_plan),
                        elem_id="regional_region_table",
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

                    with gr.Row(elem_id="regional_canvas_toolbar"):
                        snap_to_grid = gr.Checkbox(
                            label="Snap to grid",
                            value=True,
                            container=False,
                            elem_id="regional_snap_to_grid",
                        )
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
                        canvas_width = gr.Slider(
                            64,
                            2048,
                            value=1024,
                            step=int(shared.opts.res_step),
                            label="Canvas width",
                            elem_id="regional_width",
                        )
                        canvas_height = gr.Slider(
                            64,
                            2048,
                            value=1024,
                            step=int(shared.opts.res_step),
                            label="Canvas height",
                            elem_id="regional_height",
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
                    with gr.Row(equal_height=False, elem_id="regional_output_row"):
                        with gr.Column(scale=4, min_width=360):
                            output_panel = create_output_panel(
                                "regional",
                                shared.opts.outdir_txt2img_samples,
                                toprow,
                            )
                        with gr.Column(scale=2, min_width=260, elem_id="regional_output_tools"):
                            with gr.Row():
                                validate_button = gr.Button("Validate", variant="secondary")
                                preview_button = gr.Button("Preview masks", variant="secondary")
                            validation_status = gr.Markdown(
                                "✓ Plan is structurally valid.",
                                elem_id="regional_validation_status",
                            )
                            with gr.Accordion("Mask preview", open=False):
                                mask_preview = gr.Image(
                                    label="Compiled mask preview",
                                    type="pil",
                                    interactive=False,
                                    height=240,
                                )
                                mask_diagnostics = gr.Markdown(
                                    "Run preview to inspect coverage.",
                                    elem_id="regional_mask_summary",
                                )
                            with gr.Accordion("Scripts", open=False, elem_id="regional_script_container"):
                                scripts.scripts_regional.prepare_ui()
                                regional_script_inputs = scripts.scripts_regional.setup_ui(
                                    elem_id="regional_script_list"
                                )

                    with gr.Accordion("Generation controls", open=True):
                        with gr.Row():
                            engine_choice = gr.Dropdown(
                                choices=initial_engine_choices,
                                value="auto",
                                label="Regional engine",
                                interactive=initial_engine_interactive,
                                scale=3,
                            )
                            seed = gr.Number(value=-1, precision=0, label="Seed", scale=1)
                        with gr.Accordion("Seed extras", open=False):
                            with gr.Row():
                                subseed = gr.Number(
                                    value=-1,
                                    precision=0,
                                    label="Variation seed",
                                )
                                subseed_strength = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=0.0,
                                    step=0.01,
                                    label="Variation strength",
                                )
                            with gr.Row():
                                seed_resize_from_width = gr.Number(
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    maximum=2048,
                                    label="Resize seed from width",
                                )
                                seed_resize_from_height = gr.Number(
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    maximum=2048,
                                    label="Resize seed from height",
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
                            visible=bool(initial_required_fallbacks) or initial_capability_report is None,
                            interactive=bool(initial_required_fallbacks) or initial_capability_report is None,
                            info=(
                                "Required before generation can accept: "
                                + "; ".join(initial_required_fallbacks)
                                if initial_required_fallbacks
                                else "Allows compatibility fallbacks if the selected model requires them."
                            ),
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
                        with gr.Accordion("Hires", open=False):
                            hires_enabled = gr.Checkbox(
                                label="Enable high-resolution pass",
                                value=False,
                                elem_id="regional_hr-checkbox",
                            )
                            with gr.Row():
                                hires_upscaler = gr.Dropdown(
                                    choices=list(
                                        dict.fromkeys(
                                            [
                                                *shared.latent_upscale_modes,
                                                *[upscaler.name for upscaler in shared.sd_upscalers],
                                            ]
                                        )
                                    ),
                                    value=shared.latent_upscale_default_mode,
                                    label="Upscaler",
                                )
                                hires_steps = gr.Slider(
                                    0,
                                    150,
                                    value=0,
                                    step=1,
                                    label="Hires steps",
                                )
                                hires_denoising_strength = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=0.6,
                                    step=0.05,
                                    label="Denoising strength",
                                )
                            with gr.Row():
                                hires_scale = gr.Slider(
                                    1.0,
                                    4.0,
                                    value=2.0,
                                    step=0.05,
                                    label="Upscale by",
                                )
                                hires_width = gr.Slider(
                                    0,
                                    4096,
                                    value=0,
                                    step=int(shared.opts.res_step),
                                    label="Resize width to",
                                )
                                hires_height = gr.Slider(
                                    0,
                                    4096,
                                    value=0,
                                    step=int(shared.opts.res_step),
                                    label="Resize height to",
                                )
                            with gr.Row():
                                hires_distilled_cfg_scale = gr.Slider(
                                    1.0,
                                    24.0,
                                    value=3.0,
                                    step=0.5,
                                    label="Hires Distilled CFG scale",
                                )
                                hires_cfg_scale = gr.Slider(
                                    1.0,
                                    24.0,
                                    value=6.0,
                                    step=0.5,
                                    label="Hires CFG scale",
                                )
                        with gr.Accordion("Refiner", open=False):
                            refiner_enabled = gr.Checkbox(
                                label="Enable refiner",
                                value=False,
                            )
                            refiner_checkpoint = gr.Dropdown(
                                choices=_refiner_checkpoint_choices(),
                                value="",
                                label="Refiner checkpoint",
                            )
                            with gr.Row():
                                refiner_switch_mode = gr.Dropdown(
                                    choices=[
                                        ("Fraction of steps", "steps"),
                                        ("Sigma threshold", "sigma"),
                                    ],
                                    value="steps",
                                    label="Switch mode",
                                )
                                refiner_switch_at = gr.Slider(
                                    0.025,
                                    1.0,
                                    value=0.8,
                                    step=0.025,
                                    label="Switch at",
                                )
                                refiner_cfg_scale = gr.Slider(
                                    0.0,
                                    24.0,
                                    value=0.0,
                                    step=0.5,
                                    label="Refiner CFG scale",
                                    info="Uses the base CFG below 1.",
                                )
                            refiner_policy = gr.Dropdown(
                                choices=[
                                    (
                                        "Global refiner",
                                        "global_refine",
                                    ),
                                    ("Disabled", "disabled"),
                                ],
                                value="global_refine",
                                label="Regional behavior",
                            )
                    plan_hash_display = gr.Markdown(f"`{plan_hash(initial_plan)}`", label="Plan hash")

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
            region_table,
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
            subseed,
            subseed_strength,
            seed_resize_from_width,
            seed_resize_from_height,
            engine_choice,
            hires_enabled,
            hires_upscaler,
            hires_steps,
            hires_denoising_strength,
            hires_scale,
            hires_width,
            hires_height,
            hires_distilled_cfg_scale,
            hires_cfg_scale,
            refiner_enabled,
            refiner_checkpoint,
            refiner_switch_mode,
            refiner_switch_at,
            refiner_cfg_scale,
            refiner_policy,
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
        live_state_outputs = [
            plan_bridge,
            last_valid_plan,
            validation_status,
            plan_hash_display,
            raw_plan,
        ]
        canvas_live_outputs = [*live_state_outputs, canvas_preview]
        region_live_outputs = [
            *canvas_live_outputs,
            region_list,
            region_table,
            *geometry_components,
        ]
        geometry_live_outputs = [
            *canvas_live_outputs,
            geometry_history,
            geometry_future,
            undo_geometry_button,
            redo_geometry_button,
        ]

        def live_state(snapshot):
            return snapshot[0], snapshot[1], snapshot[3], snapshot[4], snapshot[5]

        def canvas_live_state(snapshot):
            return (*live_state(snapshot), snapshot[6])

        def region_live_state(snapshot):
            return (*canvas_live_state(snapshot), snapshot[7], snapshot[8], *snapshot[56:67])

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
            return (*result[:9], *result[40:])

        def canvas_action(plan_json, selected_id, width, height):
            snapshot = _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (update_canvas(plan, width, height), selected),
            )
            return canvas_live_state(snapshot)

        def generation_action(
            plan_json,
            selected_id,
            sampler_name,
            scheduler_name,
            step_count,
            cfg,
            count,
            size,
            base_seed,
            variation_seed,
            variation_strength,
            resize_seed_width,
            resize_seed_height,
            enable_hires,
            hires_upscaler_name,
            hires_step_count,
            hires_denoising,
            hires_resize_scale,
            hires_resize_width,
            hires_resize_height,
            hires_distilled_cfg,
            hires_cfg,
            enable_refiner,
            selected_refiner,
            refiner_mode,
            refiner_threshold,
            refiner_cfg,
            selected_refiner_policy,
        ):
            snapshot = _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (
                    update_refiner_policy(
                        update_generation_options(
                            plan,
                            sampler=sampler_name,
                            scheduler=scheduler_name,
                            steps=int(step_count),
                            cfg_scale=float(cfg),
                            batch_count=int(count),
                            batch_size=int(size),
                            seed=int(base_seed),
                            subseed=int(variation_seed),
                            subseed_strength=float(variation_strength),
                            seed_resize_from_width=int(resize_seed_width),
                            seed_resize_from_height=int(resize_seed_height),
                            hires_enabled=bool(enable_hires),
                            hires_upscaler=str(hires_upscaler_name),
                            hires_steps=int(hires_step_count),
                            hires_denoising_strength=float(hires_denoising),
                            hires_scale=float(hires_resize_scale),
                            hires_width=int(hires_resize_width),
                            hires_height=int(hires_resize_height),
                            hires_distilled_cfg_scale=float(hires_distilled_cfg),
                            hires_cfg_scale=float(hires_cfg),
                            refiner_enabled=bool(enable_refiner),
                            refiner_checkpoint=str(selected_refiner or ""),
                            refiner_switch_mode=str(refiner_mode),
                            refiner_switch_at=float(refiner_threshold),
                            refiner_cfg_scale=float(refiner_cfg),
                        ),
                        selected_refiner_policy,
                    ),
                    selected,
                ),
            )
            return live_state(snapshot)

        def engine_action(plan_json, selected_id, requested):
            snapshot = _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (update_engine_request(plan, requested), selected),
            )
            return live_state(snapshot)

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

            return region_live_state(_mutate(plan_json, selected_id, mutation))

        def region_text_action(plan_json, selected_id, field, value):
            snapshot = _mutate(
                plan_json,
                selected_id,
                lambda plan, selected: (
                    update_region(plan, selected, **{field: value}),
                    selected,
                )
                if selected
                else (_raise_no_selection()),
            )
            state = (
                snapshot[0],
                snapshot[1],
                snapshot[3],
                snapshot[4],
                snapshot[5],
            )
            if field == "name":
                return (*state, snapshot[6], snapshot[7], snapshot[8])
            return state

        def rectangle_action(plan_json, selected_id, history, future, x, y, width, height):
            result = _mutate_geometry(
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
            return (*canvas_live_state(result), *result[-4:])

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

        region_selection_bridge.input(
            _select_region,
            inputs=[last_valid_plan, region_selection_bridge],
            outputs=[
                selected_region_id,
                region_list,
                canvas_preview,
                region_table,
                *selected_components,
                duplicate_button,
                delete_button,
                move_up_button,
                move_down_button,
            ],
            show_progress=False,
        )
        snap_to_grid.input(
            fn=None,
            inputs=[snap_to_grid],
            outputs=[],
            js="""
                (enabled) => {
                    const root = typeof gradioApp === "function" ? gradioApp() : document;
                    const canvas = root.querySelector("#regional_canvas");
                    if (canvas) canvas.dataset.nocturneSnapGrid = enabled ? "true" : "false";
                    return [];
                }
            """,
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
                    const preview = canvas?.querySelector(".nocturne-layout-preview");
                    if (preview) {
                        canvas.dataset.nocturneZoom = String(zoom);
                        if (typeof window.nocturneFitRegionalCanvas === "function") {
                            window.nocturneFitRegionalCanvas();
                        }
                    }
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
                outputs=canvas_live_outputs,
                show_progress=False,
                trigger_mode="always_last",
            )

        generation_inputs = [
            sampler,
            scheduler,
            steps,
            cfg_scale,
            batch_count,
            batch_size,
            seed,
            subseed,
            subseed_strength,
            seed_resize_from_width,
            seed_resize_from_height,
            hires_enabled,
            hires_upscaler,
            hires_steps,
            hires_denoising_strength,
            hires_scale,
            hires_width,
            hires_height,
            hires_distilled_cfg_scale,
            hires_cfg_scale,
            refiner_enabled,
            refiner_checkpoint,
            refiner_switch_mode,
            refiner_switch_at,
            refiner_cfg_scale,
            refiner_policy,
        ]
        for component in generation_inputs:
            component.input(
                generation_action,
                inputs=[last_valid_plan, selected_region_id, *generation_inputs],
                outputs=live_state_outputs,
                show_progress=False,
                trigger_mode="always_last",
            ).then(
                _engine_preflight_controls,
                inputs=[engine_choice, last_valid_plan],
                outputs=[
                    engine_cost_warning,
                    fallback_acknowledgement,
                    accepted_fallbacks,
                    toprow.submit,
                ],
                show_progress=False,
            )
        engine_choice.input(
            engine_action,
            inputs=[last_valid_plan, selected_region_id, engine_choice],
            outputs=live_state_outputs,
            show_progress=False,
        ).then(
            _engine_preflight_controls,
            inputs=[engine_choice, last_valid_plan],
            outputs=[
                engine_cost_warning,
                fallback_acknowledgement,
                accepted_fallbacks,
                toprow.submit,
            ],
            show_progress=False,
        )
        from modules import call_queue

        fallback_acknowledgement.input(
            _accepted_fallback_controls,
            inputs=[
                engine_choice,
                fallback_acknowledgement,
                last_valid_plan,
            ],
            outputs=[accepted_fallbacks, toprow.submit],
            show_progress=False,
        )

        generation_event = dict(
            fn=call_queue.wrap_gradio_gpu_call(
                _regional_generate,
                extra_outputs=[None, None, "", "", gr.skip(), gr.skip()],
            ),
            _js="submit_regional",
            inputs=[
                generation_task,
                last_valid_plan,
                accepted_fallbacks,
                *regional_script_inputs,
            ],
            outputs=[
                output_panel.gallery,
                output_panel.player,
                output_panel.generation_info,
                output_panel.infotext,
                engine_choice,
                capability_status,
                output_panel.html_log,
            ],
            show_progress=False,
        )
        toprow.prompt.submit(**generation_event)
        toprow.submit.click(**generation_event)

        region_name.input(
            lambda plan, selected, value: region_text_action(plan, selected, "name", value),
            inputs=[last_valid_plan, selected_region_id, region_name],
            outputs=[
                plan_bridge,
                last_valid_plan,
                validation_status,
                plan_hash_display,
                raw_plan,
                canvas_preview,
                region_list,
                region_table,
            ],
            show_progress=False,
            trigger_mode="always_last",
        )
        local_positive.input(
            lambda plan, selected, value: region_text_action(plan, selected, "positive", value),
            inputs=[last_valid_plan, selected_region_id, local_positive],
            outputs=[plan_bridge, last_valid_plan, validation_status, plan_hash_display, raw_plan],
            show_progress=False,
            trigger_mode="always_last",
        )
        local_negative.input(
            lambda plan, selected, value: region_text_action(plan, selected, "negative", value),
            inputs=[last_valid_plan, selected_region_id, local_negative],
            outputs=[plan_bridge, last_valid_plan, validation_status, plan_hash_display, raw_plan],
            show_progress=False,
            trigger_mode="always_last",
        )

        region_inputs = selected_editor_components
        for component in (
            region_enabled,
            region_locked,
            region_hidden,
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
        ):
            component.input(
                region_action,
                inputs=[last_valid_plan, selected_region_id, *region_inputs],
                outputs=region_live_outputs,
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
                outputs=geometry_live_outputs,
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

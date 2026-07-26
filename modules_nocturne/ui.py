"""Native Nocturne UI surfaces."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import tempfile
from uuid import UUID

import gradio as gr

from modules import scripts, sd_samplers, sd_schedulers, shared, ui_toprow
from modules_nocturne.regional.editor import (
    add_region,
    delete_region,
    duplicate_region,
    initial_editor_plan,
    load_valid_editor_plan,
    move_region,
    region_choices,
    render_layout_svg,
    render_mask_preview,
    selected_region,
    update_canvas,
    update_engine_request,
    update_generation_options,
    update_global_prompts,
    update_region,
    validation_markdown,
)
from modules_nocturne.regional.capabilities import capability_service
from modules_nocturne.regional.errors import PlanError, PlanValidationError
from modules_nocturne.regional.model import SeedMode
from modules_nocturne.regional.serialization import canonical_json, plan_hash
from modules_nocturne.regional.project import load_project, save_project
from modules_nocturne.regional.validation import validate_plan


def _selected_updates(plan, selected_id):
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
    uuid_text = f"`{region.id}`" if region else "No region selected."
    return (*updates, gr.update(value=uuid_text))


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
    return (*common, *plan_controls, *_selected_updates(plan, selected), *_operation_updates(selected))


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


def _select_region(plan_json, selected_id):
    plan = load_valid_editor_plan(plan_json)
    region = selected_region(plan, selected_id)
    selected = str(region.id) if region else None
    return (
        selected,
        render_layout_svg(plan, selected),
        *_selected_updates(plan, selected),
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


def _save_project_file(plan_json):
    plan = load_valid_editor_plan(plan_json)
    directory = Path(tempfile.mkdtemp(prefix="nocturne-project-"))
    destination = directory / "regional.nocturne-region.json"
    save_project(destination, plan)
    return str(destination)


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
    return gr.update(choices=choices, value=current, interactive=interactive), details


def _engine_component_update(current):
    choices, interactive, _ = _capability_values()
    if current not in choices:
        choices = [*choices, current]
    return gr.update(choices=choices, value=current, interactive=interactive)


def create_regional_interface(create_output_panel: Callable, *, head: str | None = None) -> gr.Blocks:
    """Create the canonical-plan Regional authoring workspace."""

    initial_plan = initial_editor_plan()
    initial_json = canonical_json(initial_plan)
    initial_engine_choices, initial_engine_interactive, initial_capability_status = _capability_values()

    with gr.Blocks(analytics_enabled=False, head=head) as regional_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, id_part="regional")
        toprow.submit.value = "Generation unavailable"
        toprow.submit.interactive = False

        with gr.Column(elem_id="regional_workspace"):
            gr.HTML(
                """
                <style>
                    #regional_canvas svg { display:block; width:100%; min-height:20rem; max-height:34rem; }
                    #regional_canvas .nocturne-layout-preview { border-radius:var(--radius-lg); overflow:hidden; }
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
                        gr.Markdown(
                            "LoRA and extra-network tags affect the whole generation, including tags written in a local prompt."
                        )

                with gr.Column(scale=5, min_width=420):
                    canvas_preview = gr.HTML(
                        value=render_layout_svg(initial_plan),
                        elem_id="regional_canvas",
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
                        with gr.Row():
                            load_project_button = gr.Button("Open project")
                            save_project_button = gr.Button("Save project")
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

        selected_components = [
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
            region_uuid,
        ]
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
            dimension.release(
                canvas_action,
                inputs=[last_valid_plan, selected_region_id, canvas_width, canvas_height],
                outputs=full_outputs,
                show_progress=False,
            )

        generation_inputs = [sampler, scheduler, steps, cfg_scale, batch_count, batch_size, seed]
        for component in generation_inputs:
            event = component.release if isinstance(component, gr.Slider) else component.input
            event(
                generation_action,
                inputs=[last_valid_plan, selected_region_id, *generation_inputs],
                outputs=full_outputs,
                show_progress=False,
            )
        engine_choice.input(
            engine_action,
            inputs=[last_valid_plan, selected_region_id, engine_choice],
            outputs=full_outputs,
            show_progress=False,
        )
        refresh_capabilities.click(
            _capability_controls,
            inputs=[engine_choice],
            outputs=[engine_choice, capability_status],
            show_progress=False,
        )

        region_inputs = selected_components[:-1]
        for component in region_inputs:
            event = component.release if isinstance(component, gr.Slider) else component.input
            event(
                region_action,
                inputs=[last_valid_plan, selected_region_id, *region_inputs],
                outputs=full_outputs,
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
        save_project_button.click(
            _save_project_file,
            inputs=[last_valid_plan],
            outputs=[project_download],
        )

    return regional_interface


def _raise_no_selection():
    raise PlanError("editor.region.selection_required", "$.regions", "Select a region first")

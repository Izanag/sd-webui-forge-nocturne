"""Native Nocturne UI surfaces."""

from collections.abc import Callable

import gradio as gr

from modules import shared


def create_regional_interface(create_output_panel: Callable, *, head: str | None = None) -> gr.Blocks:
    """Create the inert Regional workspace without initializing generation state."""

    with gr.Blocks(analytics_enabled=False, head=head) as regional_interface:
        with gr.Column(elem_id="regional_workspace"):
            gr.HTML(
                """
                <section class="nocturne-regional-intro" aria-labelledby="regional_workspace_title">
                    <h2 id="regional_workspace_title">Regional</h2>
                    <p>Compose prompts and spatial regions in a dedicated generation workspace.</p>
                    <p role="status"><strong>Generation is not available yet.</strong> Regional controls will appear as supported runtime components become available.</p>
                </section>
                """,
                elem_id="regional_workspace_status",
            )

            with gr.Row(elem_id="regional_prompt_preview"):
                gr.Textbox(
                    label="Global prompt",
                    placeholder="Regional prompt editing is not available yet",
                    interactive=False,
                    lines=4,
                    elem_id="regional_global_prompt",
                )
                gr.Textbox(
                    label="Global negative prompt",
                    placeholder="Regional prompt editing is not available yet",
                    interactive=False,
                    lines=4,
                    elem_id="regional_global_negative_prompt",
                )

            create_output_panel("regional", shared.opts.outdir_txt2img_samples)

    return regional_interface

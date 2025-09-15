import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import diffusers as df
    from diffusers import StableDiffusionPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )
    from pprint import pprint
    import marimo as mo
    import numpy as np


@app.cell
def _():
    pprint(f"diffusers version: {df.__version__}")
    return


@app.cell
def _():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return (pipe,)


@app.cell
def _():
    mo.md(
        r"""
    ## Stable Diffusion Pipline Components
    - 
    """
    )
    return


@app.cell
def _(pipe):
    pprint(pipe.unet)
    return


@app.cell
def _():
    mo.md(r"""## Unet Structure""")
    return


@app.cell
def _(pipe):
    pprint(pipe.unet)
    return


@app.cell
def _():
    mo.md(r"""## VAE Structure""")
    return


@app.cell
def _(pipe):
    print(pipe.vae)
    return


@app.cell
def _():
    mo.md(r"""## text encoder""")
    return


@app.cell
def _(pipe):
    print(pipe.text_encoder)
    return


@app.cell
def _():
    mo.md(r"""## Feature extractor""")
    return


@app.cell
def _(pipe):
    print(pipe.feature_extractor)
    return


@app.cell
def _():
    image = np.random.randn(2,3)
    print(image)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

"""InstantStyle Gradio application for Flux and SDXL style transfer."""

import gc
import logging
import os
import random
import tempfile

import gradio as gr
import numpy as np
import spaces
import torch
import torch.cuda
from gradio_client import Client, file
from PIL import Image

from exceptions import (
    ImageProcessingError,
    InferenceError,
    ModelLoadError,
    ValidationError,
)
from logging_config import setup_logging
from metrics import timed
from model_manager import get_model_manager
from validation import MAX_IMAGE_SIZE, MIN_IMAGE_SIZE, validate_inputs

# Initialize logging
logger = setup_logging(logging.INFO)

# Constants
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_gpu_memory() -> None:
    """Clear GPU memory and cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def resize_img(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image to fit within max_size while maintaining aspect ratio."""
    width, height = image.size
    scaling_factor = min(max_size / width, max_size / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)


@spaces.GPU
@timed
def process_image(
    image: Image.Image | np.ndarray | None,
    prompt: str,
    scale: float,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, int]:
    """Process an image using IP-Adapter style transfer.

    Args:
        image: Input style image
        prompt: Text prompt for generation
        scale: IP-Adapter influence scale (0.0-1.0)
        seed: Random seed
        randomize_seed: Whether to randomize the seed
        width: Output width
        height: Output height
        progress: Gradio progress tracker

    Returns:
        Tuple of (generated image, seed used)

    Raises:
        gr.Error: On validation or processing errors
    """
    clear_gpu_memory()

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Early return for None image (before validation for UX)
    if image is None:
        return None, seed

    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    try:
        validate_inputs(image, prompt, scale, width, height)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise gr.Error(str(e)) from None

    image = resize_img(image)

    try:
        logger.info(f"Processing image: prompt='{prompt}', scale={scale}, seed={seed}")
        ip_model = get_model_manager().get_ip_model()

        result = ip_model.generate(
            pil_image=image,
            prompt=prompt,
            scale=scale,
            width=width,
            height=height,
            seed=seed,
        )
        logger.info("Image generation completed successfully")
        clear_gpu_memory()
        return result[0], seed

    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        raise gr.Error("Failed to load model. Please try again later.") from None
    except (ImageProcessingError, InferenceError) as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise gr.Error(f"Image processing failed: {e}") from None
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory", exc_info=True)
        clear_gpu_memory()
        raise gr.Error("GPU out of memory. Try reducing image size or try again.") from None
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Connection error: {e}", exc_info=True)
        raise gr.Error("Failed to connect to model service. Please try again.") from None
    except Exception as e:
        logger.error(f"Unexpected error during image processing: {e}", exc_info=True)
        clear_gpu_memory()
        raise gr.Error("An unexpected error occurred. Please try again.") from None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested."""
    if randomize_seed:
        seed = random.randint(0, 2000)
    return seed


@timed
def create_image_sdxl(
    image_pil: Image.Image | None,
    prompt: str,
    n_prompt: str,
    scale: float,
    control_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    target: str = "Load only style blocks",
) -> str | None:
    """Create an image using SDXL via remote API.

    Args:
        image_pil: Input style image
        prompt: Text prompt
        n_prompt: Negative prompt
        scale: IP-Adapter scale
        control_scale: ControlNet scale
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed
        target: Style loading mode

    Returns:
        Path to generated image or None on failure
    """
    if image_pil is None:
        logger.warning("SDXL: No image provided")
        return None

    temp_file = None
    try:
        # Use secure temporary file to avoid race conditions
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        image_pil.save(temp_path, format="PNG")
        logger.info(f"SDXL: Saved temp image to {temp_path}")

        client = Client("Hatman/InstantStyle")
        result = client.predict(
            image_pil=file(temp_path),
            prompt=prompt,
            n_prompt=n_prompt,
            scale=1,
            control_scale=control_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            target=target,
            api_name="/create_image",
        )
        logger.info("SDXL: Image generation completed")
        return result

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"SDXL connection error: {e}")
        raise gr.Error("Failed to connect to SDXL service. Please try again.") from None
    except Exception as e:
        logger.error(f"SDXL error: {e}", exc_info=True)
        raise gr.Error(f"SDXL generation failed: {e}") from None
    finally:
        # Clean up temporary file
        if temp_file is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # Best effort cleanup


# UI CSS
css = """
::-webkit-scrollbar {
    display: none;
    }

#component-0 {
    max-width: 900px;
    margin: 0 auto;
    }

.center-markdown {
    text-align: center !important;
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    }

.gradio-row {
    display: flex !important;
    gap: 1rem !important;
    flex-wrap: nowrap !important;
    }

.gradio-column {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    }
"""
title = r"""
<h1>InstantStyle Flux & SDXL</h1>
"""

description = r"""
<p>Two different models using the IP Adapter with InstantStyle to preserve style across text-to-image generation.</p>
"""

article = r"""
---
```bibtex
@article{wang2024instantstyle,
  title={InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation},
  author={Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2404.02733},
  year={2024}
}
```
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(title, elem_classes="center-markdown")
    gr.Markdown(description, elem_classes="center-markdown")

    with gr.Tab("FLUX"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(label="Input Image", type="pil")
                scale = gr.Slider(
                    label="Image Scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.7,
                )
                prompt = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                run_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1, min_width=300):
                result = gr.Image(label="Result")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=MIN_IMAGE_SIZE,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=MIN_IMAGE_SIZE,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )

    run_button.click(
        fn=process_image,
        inputs=[
            input_image,
            prompt,
            scale,
            seed,
            randomize_seed,
            width,
            height,
        ],
        outputs=[result, seed],
    )
    with gr.Tab("SDXL"):
        with gr.Row():
            with gr.Column():
                image_pil = gr.Image(label="Style Image", type="pil")
                target_radio = gr.Radio(
                    [
                        "Load only style blocks",
                        "Load only layout blocks",
                        "Load style+layout block",
                        "Load original IP-Adapter",
                    ],
                    value="Load only style blocks",
                    label="Style mode",
                )
                prompt_textbox = gr.Textbox(
                    label="Prompt", value="a dog, masterpiece, best quality, high quality"
                )
                scale_slider_sdxl = gr.Slider(
                    minimum=0, maximum=2.0, step=0.01, value=1.0, label="Scale"
                )

                with gr.Accordion(open=False, label="Advanced Options"):
                    control_scale_slider = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        label="Controlnet conditioning scale",
                    )
                    n_prompt_textbox = gr.Textbox(
                        label="Neg Prompt",
                        value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                    )
                    guidance_scale_slider = gr.Slider(
                        minimum=1, maximum=15.0, step=0.01, value=5.0, label="guidance scale"
                    )
                    num_inference_steps_slider = gr.Slider(
                        minimum=5, maximum=50.0, step=1.0, value=20, label="num inference steps"
                    )
                    seed_slider_sdxl = gr.Slider(
                        minimum=-1000000, maximum=1000000, value=1, step=1, label="Seed Value"
                    )
                    randomize_seed_checkbox_sdxl = gr.Checkbox(label="Randomize seed", value=True)
                generate_button = gr.Button("Generate Image", variant="primary")

            with gr.Column():
                generated_image = gr.Image(label="Generated Image", show_label=False)

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed_slider_sdxl, randomize_seed_checkbox_sdxl],
            outputs=seed_slider_sdxl,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image_sdxl,
            inputs=[
                image_pil,
                prompt_textbox,
                n_prompt_textbox,
                scale_slider_sdxl,
                control_scale_slider,
                guidance_scale_slider,
                num_inference_steps_slider,
                seed_slider_sdxl,
                target_radio,
            ],
            outputs=[generated_image],
        )

    gr.Markdown(article)

if __name__ == "__main__":
    logger.info("Starting InstantStyle application")
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch(share=True, show_error=True, quiet=False)

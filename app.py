"""
InstantStyle Application

This application uses two models with an IP Adapter for style-preserving text-to-image generation.
It supports both FLUX and SDXL modes in separate tabs.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from gradio_client import Client
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download, login

from pipeline_flux_ipa import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import IPAFluxAttnProcessor2_0  
from transformers import AutoProcessor, SiglipVisionModel  
from infer_flux_ipa_siglip import MLPProjModel, IPAdapter

# Device and torch data type settings
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
token = os.getenv("HF_TOKEN")
login(token=token)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """
    Randomizes the seed if randomize_seed is True.
    
    Args:
        seed (int): The original seed.
        randomize_seed (bool): Whether to randomize the seed.

    Returns:
        int: The randomized (or original) seed.
    """
    if randomize_seed:
        seed = random.randint(0, 2000)
    return seed

# Download model assets and initialize IP Adapter pipeline
image_encoder_path = "google/siglip-so400m-patch14-384"
transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16, token=token)
fluxPipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16)
ipadapter_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-IP-Adapter", filename="ip-adapter.bin")
ip_model = IPAdapter(fluxPipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

def resize_img(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resizes the given image while maintaining its aspect ratio.
    
    Args:
        image (Image.Image): The input PIL image.
        max_size (int): Maximum size for width or height.

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size
    scaling_factor = min(max_size / width, max_size / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)

def create_image_sdxl(
    image_pil,
    prompt: str,
    n_prompt: str,
    scale,
    control_scale,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    target: str = "Load only style blocks",
):
    client = Client("Hatman/InstantStyle")
    result = client.predict(
            image_pil=image_pil,
            prompt=prompt,
            n_prompt=n_prompt,
            scale=1,
            control_scale=control_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            target=target,
            api_name="/create_image"
    )
    
    return result
    
@spaces.GPU
def process_image(
    image,
    prompt: str,
    scale,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Processes the input image and generates a styled image using IP Adapter model.
    
    Args:
        image: The input image (as PIL Image or array).
        prompt (str): The prompt text.
        scale: The scale value.
        seed (int): The seed value.
        randomize_seed (bool): Whether to randomize the seed.
        width (int): Output image width.
        height (int): Output image height.
        progress: Gradio progress tracker.

    Returns:
        Tuple containing the generated image and the seed used.
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if image is None:
        return None, seed

    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    image = resize_img(image)
    
    result = ip_model.generate(
        pil_image=image,
        prompt=prompt,
        scale=scale,
        width=width,
        height=height,
        seed=seed
    )
    
    return result[0], seed

# Content for Gradio interface
title = r"""
<h1>InstantStyle-Flux</h1>
"""

description = r"""
<p>Two different models using the IP Adapter to preserve style across text-to-image generation.</p>
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
with gr.Blocks() as demo: 
    # Custom CSS for layout and appearance 
    gr.HTML(""" <style> ::-webkit-scrollbar { display: none; } #component-0 { max-width: 900px; margin: 0 auto; } 
            .center-markdown { text-align: center !important; display: flex !important; justify-content: center !important; width: 100% !important; } .gradio-row { display: flex !important; gap: 1rem !important; flex-wrap: nowrap !important; } 
            .gradio-column { flex: 1 1 0 !important; min-width: 0 !important; } </style>""")
    # Display title and description in center-aligned Markdown
    gr.Markdown(title, elem_classes="center-markdown")
    gr.Markdown(description, elem_classes="center-markdown")

    # FLUX Tab
    with gr.Tab("FLUX"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(label="Input Image", type="pil")
                scale_slider = gr.Slider(
                    label="Image Scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.7,
                )
                prompt_input = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt",
                )
                run_button = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1, min_width=300):
                result_image = gr.Image(label="Result")
        
        with gr.Accordion("Advanced Settings", open=False):
            seed_slider = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )
            randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                width_slider = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                height_slider = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

        run_button.click(
            fn=process_image,
            inputs=[
                input_image,
                prompt_input,
                scale_slider,
                seed_slider,
                randomize_seed_checkbox,
                width_slider,
                height_slider,
            ],
            outputs=[result_image, seed_slider],
        )

    # SDXL Tab
    with gr.Tab("SDXL"):
        with gr.Row():
            with gr.Column():
                image_pil = gr.Image(label="Style Image", type="pil")
                target_radio = gr.Radio(
                    ["Load only style blocks", "Load only layout blocks", "Load style+layout block", "Load original IP-Adapter"],
                    value="Load only style blocks",
                    label="Style mode"
                )
                prompt_textbox = gr.Textbox(
                    label="Prompt",
                    value="a cat, masterpiece, best quality, high quality"
                )
                scale_slider_sdxl = gr.Slider(
                    minimum=0,
                    maximum=2.0,
                    step=0.01,
                    value=1.0,
                    label="Scale"
                )
                
                with gr.Accordion(open=False, label="Advanced Options"):
                    control_scale_slider = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        label="Controlnet conditioning scale"
                    )
                    n_prompt_textbox = gr.Textbox(
                        label="Neg Prompt",
                        value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
                    )
                    guidance_scale_slider = gr.Slider(
                        minimum=1,
                        maximum=15.0,
                        step=0.01,
                        value=5.0,
                        label="guidance scale"
                    )
                    num_inference_steps_slider = gr.Slider(
                        minimum=5,
                        maximum=50.0,
                        step=1.0,
                        value=20,
                        label="num inference steps"
                    )
                    seed_slider_sdxl = gr.Slider(
                        minimum=-1000000,
                        maximum=1000000,
                        value=1,
                        step=1,
                        label="Seed Value"
                    )
                    randomize_seed_checkbox_sdxl = gr.Checkbox(label="Randomize seed", value=True)
                generate_button = gr.Button("Generate Image")
        
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
            outputs=[generated_image]
        )

    gr.Markdown(article)

if __name__ == "main": 
    demo.launch()
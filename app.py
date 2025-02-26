import gradio as gr
import numpy as np
import random
import torch
import spaces
from PIL import Image
import os
import torch.cuda
import gc
from gradio_client import Client, file
from pipeline_flux_ipa import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel
from infer_flux_ipa_siglip import MLPProjModel, IPAdapter
from huggingface_hub import hf_hub_download

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image_encoder_path = "google/siglip-so400m-patch14-384"
ipadapter_path = hf_hub_download(repo_id="InstantX/FLUX.1-dev-IP-Adapter", filename="ip-adapter.bin")

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer", 
    torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    transformer=transformer, 
    torch_dtype=torch.bfloat16
)

ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def resize_img(image, max_size=1024):
    width, height = image.size
    scaling_factor = min(max_size / width, max_size / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)
    
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
    clear_gpu_memory()
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
    clear_gpu_memory()
    return result[0], seed

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, 2000)
    return seed
    
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
    try:
        image_pil.save("./tmp.png", format="PNG")
        client = Client("Hatman/InstantStyle")
        result = client.predict(
            image_pil=file("./tmp.png"),
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
        
    except Exception as e:
        print(f"Error in create_image_sdxl: {str(e)}")
        return None
    
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
                input_image = gr.Image(
                    label="Input Image",
                    type="pil"
                )
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
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
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
                    ["Load only style blocks", "Load only layout blocks", "Load style+layout block", "Load original IP-Adapter"],
                    value="Load only style blocks",
                    label="Style mode"
                )
                prompt_textbox = gr.Textbox(
                    label="Prompt",
                    value="a dog, masterpiece, best quality, high quality"
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
            outputs=[generated_image]
        )

    gr.Markdown(article)

if __name__ == "__main__":  
    demo.launch(
        share=True,  
        show_error=True,  
        quiet=False
    )
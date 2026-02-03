"""IP-Adapter inference module for Flux models with SigLIP image encoder."""

import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel

from attention_processor import IPAFluxAttnProcessor2_0
from exceptions import ImageProcessingError, ModelLoadError
from metrics import timed

logger = logging.getLogger("instant_style.ip_adapter")


def resize_img(
    input_image: Image.Image,
    max_side: int = 1280,
    min_side: int = 1024,
    size: tuple[int, int] | None = None,
    pad_to_max_side: bool = False,
    mode: int = Image.BILINEAR,
    base_pixel_number: int = 64,
) -> Image.Image:
    """Resize image while maintaining aspect ratio.

    Args:
        input_image: PIL Image to resize
        max_side: Maximum side length
        min_side: Minimum side length
        size: Optional explicit size (width, height)
        pad_to_max_side: Whether to pad to max_side square
        mode: PIL resampling mode
        base_pixel_number: Align dimensions to this multiple

    Returns:
        Resized PIL Image
    """
    import numpy as np

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(
            input_image
        )
        input_image = Image.fromarray(res)
    return input_image


class MLPProjModel(nn.Module):
    """MLP projection model for image embeddings."""

    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """Project identity embeddings to cross-attention dimension.

        Args:
            id_embeds: Input identity embeddings

        Returns:
            Projected and normalized embeddings
        """
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class IPAdapter:
    """IP-Adapter for style transfer with Flux models."""

    def __init__(
        self,
        sd_pipe: object,
        image_encoder_path: str,
        ip_ckpt: str,
        device: str,
        num_tokens: int = 4,
    ) -> None:
        """Initialize IP-Adapter.

        Args:
            sd_pipe: Flux pipeline instance
            image_encoder_path: Path to SigLIP image encoder
            ip_ckpt: Path to IP-Adapter checkpoint
            device: Device to use (cuda/cpu)
            num_tokens: Number of image tokens

        Raises:
            ModelLoadError: If model loading fails
        """
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        logger.info(f"Initializing IPAdapter on device: {device}")
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # Load image encoder
        logger.info(f"Loading image encoder from: {image_encoder_path}")
        try:
            self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(
                self.device, dtype=torch.bfloat16
            )
            self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load image encoder from {image_encoder_path}",
                model_path=image_encoder_path,
                cause=e,
            ) from e

        # Image projection model
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()
        logger.info("IPAdapter initialization complete")

    def init_proj(self) -> MLPProjModel:
        """Initialize the image projection model.

        Returns:
            Initialized MLPProjModel
        """
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim,
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)

        return image_proj_model

    def set_ip_adapter(self) -> None:
        """Configure IP-Adapter attention processors on the transformer."""
        transformer = self.pipe.transformer
        ip_attn_procs = {}
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks.") or name.startswith(
                "single_transformer_blocks"
            ):
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=transformer.config.num_attention_heads
                    * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = transformer.attn_processors[name]

        transformer.set_attn_processor(ip_attn_procs)

    def load_ip_adapter(self) -> None:
        """Load IP-Adapter weights from checkpoint.

        Raises:
            ModelLoadError: If checkpoint loading fails
        """
        if not os.path.exists(self.ip_ckpt):
            raise ModelLoadError(
                f"IP-Adapter checkpoint not found: {self.ip_ckpt}",
                model_path=self.ip_ckpt,
            )

        logger.info(f"Loading IP-Adapter weights from: {self.ip_ckpt}")
        try:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load IP-Adapter checkpoint: {self.ip_ckpt}",
                model_path=self.ip_ckpt,
                cause=e,
            ) from e

        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = nn.ModuleList(self.pipe.transformer.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
        logger.info("IP-Adapter weights loaded successfully")

    @torch.inference_mode()
    @timed
    def get_image_embeds(
        self,
        pil_image: Image.Image | list[Image.Image] | None = None,
        clip_image_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract image embeddings using the image encoder.

        Args:
            pil_image: PIL Image or list of images
            clip_image_embeds: Pre-computed CLIP embeddings (alternative to pil_image)

        Returns:
            Image prompt embeddings tensor

        Raises:
            ImageProcessingError: If image processing fails
        """
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]

            try:
                clip_image = self.clip_image_processor(
                    images=pil_image, return_tensors="pt"
                ).pixel_values
                clip_image_embeds = self.image_encoder(
                    clip_image.to(self.device, dtype=self.image_encoder.dtype)
                ).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
            except Exception as e:
                raise ImageProcessingError(f"Failed to process image: {e}") from e
        else:
            if clip_image_embeds is None:
                raise ImageProcessingError("Either pil_image or clip_image_embeds must be provided")
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

    @timed
    def generate(
        self,
        pil_image: Image.Image | list[Image.Image] | None = None,
        clip_image_embeds: torch.Tensor | None = None,
        prompt: str | None = None,
        scale: float = 1.0,
        num_samples: int = 1,
        seed: int | None = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 24,
        **kwargs: object,
    ) -> list[Image.Image]:
        """Generate images using IP-Adapter style transfer.

        Args:
            pil_image: Style reference image(s)
            clip_image_embeds: Pre-computed CLIP embeddings (alternative to pil_image)
            prompt: Text prompt for generation
            scale: IP-Adapter scale factor (0.0-1.0)
            num_samples: Number of images to generate
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            **kwargs: Additional arguments passed to pipeline

        Returns:
            List of generated PIL Images
        """
        logger.info(
            f"Generating image: prompt='{prompt}', scale={scale}, "
            f"seed={seed}, steps={num_inference_steps}"
        )

        image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)

        images = self.pipe(
            prompt=prompt,
            image_emb=image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            scale=scale,
            **kwargs,
        ).images

        logger.info(f"Generated {len(images)} image(s)")
        return images


if __name__ == "__main__":
    from pipeline_flux_ipa import FluxPipeline
    from transformer_flux import FluxTransformer2DModel

    model_path = "black-forest-labs/FLUX.1-dev"
    image_encoder_path = "google/siglip-so400m-patch14-384"
    ipadapter_path = "./ip-adapter.bin"

    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    pipe = FluxPipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.bfloat16
    )

    ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

    image_dir = "./assets/images/2.jpg"
    image_name = image_dir.split("/")[-1]
    image = Image.open(image_dir).convert("RGB")
    image = resize_img(image)

    prompt = "a young girl"

    images = ip_model.generate(
        pil_image=image, prompt=prompt, scale=0.7, width=960, height=1280, seed=42
    )

    images[0].save(f"results/{image_name}")

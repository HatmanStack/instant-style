"""Thread-safe lazy model loading for InstantStyle."""

import logging
import threading
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from exceptions import ModelLoadError
from metrics import timed

logger = logging.getLogger("instant_style.model_manager")


class ModelManager:
    """Thread-safe lazy loader for ML models.

    Ensures models are only loaded once, even under concurrent access.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._models_lock = threading.Lock()
        self._transformer: Optional[torch.nn.Module] = None
        self._pipe: Optional[object] = None
        self._ip_model: Optional[object] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self._device}")

    @property
    def device(self) -> str:
        return self._device

    @timed
    def _load_transformer(self) -> torch.nn.Module:
        """Load the Flux transformer model."""
        from transformer_flux import FluxTransformer2DModel

        logger.info("Loading FluxTransformer2DModel...")
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            logger.info("FluxTransformer2DModel loaded successfully")
            return transformer
        except Exception as e:
            raise ModelLoadError(
                "Failed to load FluxTransformer2DModel",
                model_path="black-forest-labs/FLUX.1-dev",
                cause=e,
            ) from e

    @timed
    def _load_pipeline(self, transformer: torch.nn.Module) -> object:
        """Load the Flux pipeline."""
        from pipeline_flux_ipa import FluxPipeline

        logger.info("Loading FluxPipeline...")
        try:
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            logger.info("FluxPipeline loaded successfully")
            return pipe
        except Exception as e:
            raise ModelLoadError(
                "Failed to load FluxPipeline",
                model_path="black-forest-labs/FLUX.1-dev",
                cause=e,
            ) from e

    @timed
    def _load_ip_adapter(self, pipe: object) -> object:
        """Load the IP Adapter model."""
        from infer_flux_ipa_siglip import IPAdapter

        image_encoder_path = "google/siglip-so400m-patch14-384"
        logger.info("Downloading IP Adapter weights...")
        try:
            ipadapter_path = hf_hub_download(
                repo_id="InstantX/FLUX.1-dev-IP-Adapter", filename="ip-adapter.bin"
            )
        except Exception as e:
            raise ModelLoadError(
                "Failed to download IP Adapter weights",
                model_path="InstantX/FLUX.1-dev-IP-Adapter",
                cause=e,
            ) from e

        logger.info("Loading IPAdapter...")
        try:
            ip_model = IPAdapter(
                pipe, image_encoder_path, ipadapter_path, device=self._device, num_tokens=128
            )
            logger.info("IPAdapter loaded successfully")
            return ip_model
        except Exception as e:
            raise ModelLoadError(
                "Failed to initialize IPAdapter",
                model_path=ipadapter_path,
                cause=e,
            ) from e

    def get_ip_model(self) -> object:
        """Get the IP Adapter model, loading if necessary.

        Thread-safe lazy loading ensures models are loaded only once.

        Returns:
            Initialized IPAdapter instance

        Raises:
            ModelLoadError: If model loading fails
        """
        if self._ip_model is not None:
            return self._ip_model

        with self._models_lock:
            # Double-check after acquiring lock
            if self._ip_model is not None:
                return self._ip_model

            logger.info("First request - initializing models...")
            self._transformer = self._load_transformer()
            self._pipe = self._load_pipeline(self._transformer)
            self._ip_model = self._load_ip_adapter(self._pipe)
            return self._ip_model

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._ip_model is not None


# Global singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager singleton."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

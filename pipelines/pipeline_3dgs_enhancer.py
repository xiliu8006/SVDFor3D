from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.utils import BaseOutput

from pipelines.pipeline_stable_flow_video_diffusion import StableFlowVideoDiffusionPipeline
from refine_decoder import RefineDecoderModel


@dataclass
class EnhancerPipelineOutput(BaseOutput):
    """
    Args:
        frames: enhanced frames, `[T, 3, H, W]` in `[0, 1]`.
        diffusion_frames: VAE decoded video diffusion output, `[T, 3, H, W]` in `[0, 1]`, only if requested.
    """

    frames: torch.Tensor
    diffusion_frames: Optional[torch.Tensor] = None


class Enhancer3DGSPipeline(StableFlowVideoDiffusionPipeline):
    """3DGS-Enhancer: a pose conditioned video diffusion model whose latents are decoded by the refinement decoder."""

    def __init__(self, vae, image_encoder, unet, scheduler, feature_extractor, refine_decoder: RefineDecoderModel):
        super().__init__(
            vae=vae, image_encoder=image_encoder, unet=unet, scheduler=scheduler, feature_extractor=feature_extractor
        )
        self.register_modules(refine_decoder=refine_decoder)

    @torch.no_grad()
    def generate_latents(self, image, poses, **kwargs):
        """Video diffusion stage. Returns latents `[T, 4, H/8, W/8]`, divided by the VAE scaling factor."""
        latents = super().__call__(image, poses=poses, output_type="latent", **kwargs).frames
        return latents[0] / self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_diffusion(self, latents, image, decode_chunk_size=8):
        """VAE decodes the diffusion latents, only to inspect the intermediate result."""
        scaled = (latents * self.vae.config.scaling_factor)[None].to(self.vae.dtype)
        frames = self.decode_latents(scaled, latents.shape[0], decode_chunk_size)[0].permute(1, 0, 2, 3)
        frames = ((frames.float() / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() / 255.0
        frames = frames.to(image)
        frames[0], frames[-1] = image[0], image[-1]
        return frames

    @torch.no_grad()
    def refine_latents(self, image, latents, fusion_w=None):
        """Refinement decoder: decodes the diffusion latents, fusing features of the input frames."""
        dtype = self.refine_decoder.dtype
        output = self.refine_decoder(image.to(dtype=dtype) * 2.0 - 1.0, latents.to(dtype=dtype), fusion_w=fusion_w)
        return ((output + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(
        self,
        image,
        poses,
        height=512,
        width=960,
        decode_chunk_size=8,
        fusion_w=None,
        return_diffusion_frames=False,
        **kwargs,
    ):
        """
        Args:
            image: `[T, 3, H, W]` in `[0, 1]`: 3DGS renderings, with clean input views as the first and last frames.
            poses: camera conditioning `[T, 19]`, as built by `DummyInferenceDataset`.
            fusion_w: weight of the input frame features in the decoder; defaults to the value in the checkpoint.
            return_diffusion_frames: also return the VAE decoded diffusion output.
        """
        latents = self.generate_latents(
            image,
            poses,
            height=height,
            width=width,
            num_frames=image.shape[0],
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=127,
            **kwargs,
        )
        diffusion_frames = self.decode_diffusion(latents, image, decode_chunk_size) if return_diffusion_frames else None
        torch.cuda.empty_cache()
        return EnhancerPipelineOutput(frames=self.refine_latents(image, latents, fusion_w), diffusion_frames=diffusion_frames)

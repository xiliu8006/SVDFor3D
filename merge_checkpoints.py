"""Merges a trained video diffusion UNet and a refinement decoder into one 3DGS-Enhancer checkpoint."""
import argparse
import os

import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from pipelines.pipeline_3dgs_enhancer import Enhancer3DGSPipeline
from refine_decoder import RefineDecoderModel
from UNet3D.UNetSpatioTemporalFlowCondition import UNetSpatioTemporalFlowConditionModel

COMPONENTS = ["unet", "vae", "image_encoder", "refine_decoder"]


def build_pipeline(svd_model_path, decoder_ckpt, fusion_w, pretrained):
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained, subfolder="feature_extractor")
    scheduler = EulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained, subfolder="image_encoder", variant="fp16", torch_dtype=torch.float32
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained, subfolder="vae", variant="fp16", torch_dtype=torch.float32
    )
    unet = UNetSpatioTemporalFlowConditionModel.from_pretrained(svd_model_path, subfolder="unet")
    refine_decoder, _ = RefineDecoderModel.load_lightning_ckpt(decoder_ckpt, fusion_w=fusion_w)
    return Enhancer3DGSPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        refine_decoder=refine_decoder,
    )


def assert_same_weights(pipeline, reloaded):
    for name in COMPONENTS:
        source = getattr(pipeline, name).state_dict()
        target = getattr(reloaded, name).state_dict()
        assert source.keys() == target.keys(), f"{name}: keys differ after reload"
        for key, value in source.items():
            assert torch.equal(value, target[key]), f"{name}.{key} differs after reload"
        print(f"verified {name}: {len(source)} tensors identical after reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, required=True, help="video diffusion training output containing unet/")
    parser.add_argument("--decoder_ckpt", type=str, required=True, help="PyTorch Lightning checkpoint of the decoder")
    parser.add_argument("--fusion_w", type=float, default=0.5)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        raise FileExistsError(f"{args.output_path} already exists")

    pipeline = build_pipeline(args.svd_model_path, args.decoder_ckpt, args.fusion_w, args.pretrained)
    pipeline.save_pretrained(args.output_path, safe_serialization=True)

    assert_same_weights(pipeline, Enhancer3DGSPipeline.from_pretrained(args.output_path))
    print(f"saved to {args.output_path}")

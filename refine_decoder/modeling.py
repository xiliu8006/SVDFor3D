"""Spatial-temporal refinement decoder of 3DGS-Enhancer."""
import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _NoLightningUnpickler(pickle.Unpickler):
    """Loads a PyTorch Lightning checkpoint without importing pytorch_lightning; only the weights are used."""

    def find_class(self, module, name):
        if module.split(".")[0] == "pytorch_lightning":
            return _Stub
        return super().find_class(module, name)


_NoLightningPickle = types.ModuleType("no_lightning_pickle")
_NoLightningPickle.Unpickler = _NoLightningUnpickler
_NoLightningPickle.load = pickle.load


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        h_ = torch.bmm(v.reshape(b, c, h * w), w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = self.conv1(nonlinearity(self.norm1(x_in)))
        x = self.conv2(nonlinearity(self.norm2(x)))
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_V1(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb2(self.rdb1(x)) * 0.2 + x


class FuseBlock(nn.Module):
    """Adds a residual predicted from the LQ encoder feature and the decoder feature."""

    def __init__(self, in_ch, out_ch, num_block=2, num_grow_ch=32):
        super().__init__()
        self.encode_enc_1 = ResBlock(2 * in_ch, in_ch)
        self.encode_enc_2 = nn.Sequential(*[RRDB_V1(in_ch, num_grow_ch) for _ in range(num_block)])
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w):
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        return dec_feat + w * enc_feat


class SpatialTemporalConv(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.temporal_conv = nn.Conv3d(num_feat, num_feat, (3, 1, 1), padding=(1, 0, 0))
        self.temporal_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, num_frames):
        bt, c, h, w = x.shape
        res = x.reshape(bt // num_frames, num_frames, c, h, w).permute(0, 2, 1, 3, 4)
        res = self.temporal_conv(res).permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)
        return self.temporal_alpha * res + (1 - self.temporal_alpha) * x


class Encoder(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, in_channels, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        features = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level == 1 or i_level == 2:
                features.append(h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(nonlinearity(self.norm_out(h))), features


class VideoDecoderMix(nn.Module):
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, z_channels, num_fuse_block):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.temporal_mixing = SpatialTemporalConv(block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            temporal_mixing = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            if i_level != self.num_resolutions - 1 and i_level != 0:
                setattr(self, f"fusion_layer_{i_level}", FuseBlock(block_out, block_out, num_fuse_block))

            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                temporal_mixing.append(SpatialTemporalConv(block_in))

            up = nn.Module()
            up.block = block
            up.temporal_mixing = temporal_mixing
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, enc_fea, num_frames, fusion_w):
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.temporal_mixing(h, num_frames)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                h = self.up[i_level].temporal_mixing[i_block](h, num_frames)

            if i_level != self.num_resolutions - 1 and i_level != 0:
                fuse_layer = getattr(self, f"fusion_layer_{i_level}")
                h = fuse_layer(enc_fea[i_level - 1], h, fusion_w)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        return self.conv_out(nonlinearity(self.norm_out(h)))


class RefineDecoderModel(ModelMixin, ConfigMixin):
    """Decodes video diffusion latents into frames, fusing encoder features of the input frames."""

    @register_to_config
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        in_channels=3,
        z_channels=4,
        embed_dim=4,
        num_fuse_block=2,
        fusion_w=1.0,
    ):
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, num_res_blocks, in_channels, z_channels)
        self.decoder = VideoDecoderMix(ch, out_ch, ch_mult, num_res_blocks, z_channels, num_fuse_block)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x):
        h, features = self.encoder(x)
        return self.quant_conv(h), features

    def forward(self, lq_frames, latents, fusion_w=None):
        """Decodes video diffusion latents, fusing encoder features of the input frames.

        Args:
            lq_frames: `[T, 3, H, W]` in `[-1, 1]`; the first and last frames are clean input views.
            latents: `[T, 4, H/8, W/8]` diffusion latents divided by the VAE scaling factor. The encoder of this model
                is the SVD VAE encoder, so both share one latent space.
        """
        fusion_w = self.config.fusion_w if fusion_w is None else fusion_w
        moments, enc_fea_lq = self.encode(lq_frames)
        # the clean input views keep their own latents
        ref_latents = torch.chunk(moments, 2, dim=1)[0]
        latents = latents.clone()
        latents[0], latents[-1] = ref_latents[0], ref_latents[-1]
        del moments
        return self.decoder(self.post_quant_conv(latents.to(lq_frames)), enc_fea_lq, lq_frames.shape[0], fusion_w)

    @classmethod
    def load_lightning_ckpt(cls, ckpt_path, **config):
        """Builds the model from a PyTorch Lightning checkpoint of the decoder training."""
        ckpt = torch.load(ckpt_path, map_location="cpu", pickle_module=_NoLightningPickle)
        model = cls(**config)
        expected = model.state_dict().keys()
        state_dict = {k: v for k, v in ckpt["state_dict"].items() if k in expected}
        missing = sorted(set(expected) - set(state_dict))
        if missing:
            raise ValueError(f"{ckpt_path} misses {len(missing)} decoder keys, first: {missing[:5]}")
        model.load_state_dict(state_dict, strict=True)
        return model, ckpt.get("global_step")

#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
import copy
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
from UNet3D.UNetSpatioTemporalFlowCondition import UNetSpatioTemporalFlowConditionModel
from pipelines.pipeline_stable_flow_video_diffusion import StableFlowVideoDiffusionPipeline
from scipy.spatial.transform import Rotation as R

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class DummyDataset(Dataset):
    def __init__(self, base_folder, ref_folders, width=1024, height=576, sample_frames=26, max_ratio=3):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = base_folder
        self.ref_folders = ref_folders
        self.base_scenes = set(os.listdir(self.base_folder))
        self.ref_scenes = set(os.listdir(self.ref_folders))
        self.scenes = list(self.base_scenes.intersection(self.ref_scenes))
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.max_step = max_ratio + 1
        self.samples = {}
        self.random_samples = {}
        self.view_num = os.listdir(os.path.join(self.base_folder, self.scenes[0], 'lr'))
        self.length = 0

        for scene in self.scenes:
            final_scene_path = os.path.join(self.base_folder, scene, 'lr', '24', 'train_24')
            if not os.path.exists(final_scene_path):
                    print("this scene is not finnished: ", final_scene_path)
                    continue
            self.samples[scene] = {}
            self.random_samples[scene] = {}
            for view_num in os.listdir(os.path.join(self.base_folder, scene, 'lr')):
                scene_path = os.path.join(self.base_folder, scene, 'lr', view_num, 'train_'+ view_num)
                frames = sorted([img for img in os.listdir(scene_path)])
                if len(frames) < 50:
                    continue
                self.random_samples[scene][view_num] = copy.deepcopy(frames)
                self.samples[scene][view_num] = []
                ref_indices = [index for index, value in enumerate(frames) if '_ref' in value]
                # if abs(ref_indices[1] - ref_indices[0]) == (sample_frames - 1):
                for start, end in zip(ref_indices, ref_indices[1:] + [None]):
                    if end is not None:
                        sample = frames[start:(end+1)]
                        if len(sample) != sample_frames:
                            if len(sample) < sample_frames:
                                # print("sample length: ", len(sample))
                                if len(sample) > 10:
                                    sample = self.create_fixed_length_sample(sample, sample_frames)
                                else:
                                    continue
                            else:
                                sample = self.uniform_sample_with_fixed_count(sample, sample_frames)
                            # if start + sample_frames > len(frames):
                            #     sample = frames[-sample_frames:]
                            #     sample[0] = frame[start]
                            # elif start - end > sample_frames:
                            #     sample = self.uniform_sample_with_fixed_count(frames, sample_frames)
                            # else:
                            #     sample = frames[start: (start + sample_frames)]
                        # assert len(sample)==26, f'the scene is {scene} and view num is {view_num} and len of sample is {len(sample)} '
                        self.samples[scene][view_num].append(sample)
                        self.length = self.length + 1
                # else:
                #     for i in range(0, len(frames), sample_frames):
                #         sample = frames[i:(i+sample_frames)]
                #         if len(sample) != sample_frames:
                #             continue
                #         else:
                #             self.samples[scene][view_num].append(sample)
        self.scenes = list(self.samples.keys())
        self.colmap = {}
        self.camera = {}
        for scene in self.scenes:
            images_path = os.path.join(self.base_folder, scene, 'lr', '24', 'sparse/0_render/images.txt')
            camera_path = os.path.join(self.base_folder, scene, 'lr', '24', 'sparse/0_render/cameras.txt')
            self.colmap[scene] = self.read_colmap_images(images_path)
            self.camera[scene] = self.load_cameras_and_calculate_fov(camera_path)
    
    def create_fixed_length_sample(self, frames, target_length):
        new_sample = frames[:]
        while len(new_sample) < target_length:
            index_to_duplicate = random.randint(1, len(new_sample) - 2)
            element_to_duplicate = new_sample[index_to_duplicate]
            new_sample.insert(index_to_duplicate + 1, element_to_duplicate)
        assert len(new_sample)==target_length
        return new_sample
        
    def uniform_sample_with_fixed_count(self, lst, count):
        sampled_list = [lst[0]]
        remaining_count = count - 2
        sublist = lst[1:-1]
        interval = len(sublist) // remaining_count
        start = int((len(lst) - interval * 23) / 2)
        
        for i in range(remaining_count):
            index = (start + i * interval) % len(sublist)
            sampled_list.append(sublist[index])

        sampled_list.append(lst[-1])
        return sampled_list

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        chosen_scene = self.scenes[idx]
        while not self.samples[chosen_scene]:
            print("error scene: ", chosen_scene)
            chosen_scene = random.choice(list(self.samples.keys()))

        view_num = random.choice(list(self.samples[chosen_scene].keys()))
        # Ensure selected view has frames
        while not self.samples[chosen_scene][view_num]:
            # assert len(selected_frames)==26, f'the scene is {chosen_scene} and view num is {view_num}'
            print("before no valid view num: ", view_num)
            view_num = random.choice(list(self.samples[chosen_scene].keys()))
            print("no valid view num: ", view_num)
        mode = random.choice(['Ref_interp'])
        if mode == 'Ref_interp':
            # assert len(selected_frames)==26, f'the scene is {chosen_scene} and view num is {view_num}'
            selected_frames = random.choice(self.samples[chosen_scene][view_num])
            # if(len(selected_frames) != self.sample_frames):
            #     print("selected_frames: ", mode, len(selected_frames))
        else:
            frames = self.random_samples[chosen_scene][view_num]
            
            # step = random.randint(1, self.max_step)
            step = 1
            # Sort the frames by name
            frames.sort()
            # Ensure the selected folder has at least `sample_frames`` frames
            while len(frames) < self.sample_frames * step and step > 1:
                step -= 1
            # Randomly select a start index for frame sequence
            start_idx = random.randint(0, len(frames) - (self.sample_frames * step))
            selected_frames = frames[start_idx:(start_idx + self.sample_frames* step):step]
            if(len(selected_frames) != self.sample_frames):
                print("selected_frames: ", mode, len(selected_frames))
        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        poses = torch.empty((self.sample_frames, 19))
        # Load and process each frame
        extriniscs_list = []
        # assert len(selected_frames)==26, f'the selected_frames shape is {len(selected_frames)}'
        for i, frame_name in enumerate(selected_frames):
            # print('extriniscs: ', chosen_scene, frame_name)
            if chosen_scene in self.colmap:
                 # Check if the frame_name exists within the chosen scene
                if frame_name in self.colmap[chosen_scene]:
                    extrinsics = self.colmap[chosen_scene][frame_name]
                else:
                    if 'ref' in frame_name:
                        frame_name = frame_name.replace('_ref.png', '.png')
                    else:
                        frame_name = frame_name.replace('.png', '_ref.png')
                    if frame_name in self.colmap[chosen_scene]:
                        extrinsics = self.colmap[chosen_scene][frame_name]
                    else:
                        print(f"Frame '{frame_name}' does not exist in scene '{chosen_scene}'. and all keys are {self.colmap[chosen_scene].keys()}")
            else:
                print(f"Scene '{chosen_scene}' does not exist in colmap.")
            # extriniscs = self.colmap[chosen_scene][frame_name]
            # assert extriniscs is not None, f'extriniscs is None {chosen_scene} and frame {frame_name} and scene is {chosen_scene} and view num is {view_num}'
            extriniscs_list.append(extrinsics)
        
        poses = self.compute_T(extriniscs_list)
        poses = poses.reshape((self.sample_frames, 16))

        fov_deg = self.camera[chosen_scene]['1']['fov_horizontal']
        fov_rad = torch.tensor(fov_deg * np.pi / 180)
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )

        fov_enc = fov_enc.repeat(self.sample_frames, 1)
        T = torch.cat([poses, fov_enc], axis = 1)

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(self.base_folder, chosen_scene, 'lr', view_num, f"train_{view_num}", frame_name)
            # img = self.center_crop_to_shorter_axis(frame_path)
            # img_resized = img.resize((self.width, self.height))
            img = self.center_crop(frame_path, self.width, self.height)
            img_tensor = torch.from_numpy(np.array(img)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            # Rearrange channels if necessary
            if self.channels == 3:
                img_normalized = img_normalized.permute(
                    2, 0, 1)  # For RGB images
            elif self.channels == 1:
                img_normalized = img_normalized.mean(
                    dim=2, keepdim=True)  # For grayscale images

            condition_pixel_values[i] = img_normalized
        
        for i, frame_name in enumerate(selected_frames):
            ref_frame_name = copy.deepcopy(frame_name)
            ref_frame_name = ref_frame_name.replace('_ref', '')
            gt_frame_path = os.path.join(self.ref_folders, chosen_scene, 'images_4', ref_frame_name)

            # img = self.center_crop_to_shorter_axis(gt_frame_path)
            # img_resized = img.resize((self.width, self.height))
            img = self.center_crop(gt_frame_path, self.width, self.height)
            img_tensor = torch.from_numpy(np.array(img)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            # Rearrange channels if necessary
            if self.channels == 3:
                img_normalized = img_normalized.permute(
                    2, 0, 1)  # For RGB images
            elif self.channels == 1:
                img_normalized = img_normalized.mean(
                    dim=2, keepdim=True)  # For grayscale images
            pixel_values[i] = img_normalized

        return {'pixel_values': pixel_values, "condition_pixel_values":condition_pixel_values, "condition_T": T}
    
    def center_crop_to_shorter_axis(self, image_path):
        # Open the image file
        img = self.open_image(image_path)
        
        # Get the dimensions
        width, height = img.size
        
        # Determine the size of the square to crop to (size of the shorter dimension)
        crop_size = min(width, height)
        
        # Calculate the cropping box
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        
        # Perform the crop
        img_cropped = img.crop((left, top, right, bottom))
        
        # Return the cropped image
        return img_cropped
    
    def center_crop(self, image_path, target_width, target_height):
        # Open the image file
        img = self.open_image(image_path)
        
        # Get the dimensions
        width, height = img.size
        
        # Calculate the cropping box
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        
        # Ensure the crop box is within the image dimensions
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        # Perform the crop
        img_cropped = img.crop((left, top, right, bottom))
        
        # Return the cropped image
        return img_cropped
    
    def open_image(self, file_name):
        name, _ = os.path.splitext(file_name)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 添加你认为可能的图片格式
        for ext in possible_extensions:
            full_path = name + ext
            full_path_upper = name + ext.upper()
            if os.path.isfile(full_path):
                return Image.open(full_path)
            elif os.path.isfile(full_path_upper):
                return Image.open(full_path_upper)
        raise FileNotFoundError("No image file found for {}".format(file_name))

    def read_colmap_images(self, filename):
        images = {}
        with open(filename, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    # Verify we have at least the expected 10 elements
                    if len(parts) < 10:
                        # print(f"Skipping line {i + 1}: insufficient data.")
                        i += 1
                        continue

                    image_id = parts[0]
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = parts[8]
                    image_name = parts[9]

                    images[image_name] = {
                        'image_id': image_id,
                        'qw': qw,
                        'qx': qx,
                        'qy': qy,
                        'qz': qz,
                        'tx': tx,
                        'ty': ty,
                        'tz': tz,
                        'camera_id': camera_id
                    }

                    # Skip the next line if it's expected to contain only point data or be empty
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith('#'):
                        i += 1  # Only skip if the next line doesn't start with a comment
                i += 1
        return images


    def load_cameras_and_calculate_fov(self, filename):
        """
        Load cameras from a COLMAP cameras.txt file and calculate the field of view (FOV) for each camera.

        Args:
            filename (str): Path to the cameras.txt file.

        Returns:
            dict: A dictionary with camera IDs as keys and tuples of (horizontal FOV, vertical FOV) as values.
        """
        cameras = {}
        with open(filename, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 9:  # Check for at least enough parts for a camera entry
                        camera_id = parts[0]
                        model = parts[1]
                        width = int(parts[2])
                        height = int(parts[3])
                        params = list(map(float, parts[4:]))  # Includes focal length and other parameters

                        # Assuming the first parameter is the focal length (typical for pinhole and similar models)
                        focal_length = params[0]
                        fov_horizontal = 2 * math.atan(width / (2 * focal_length)) * (180 / math.pi)  # Convert to degrees
                        fov_vertical = 2 * math.atan(height / (2 * focal_length)) * (180 / math.pi)  # Convert to degrees

                        cameras[camera_id] = {
                            'model': model,
                            'width': width,
                            'height': height,
                            'focal_length': focal_length,
                            'fov_horizontal': fov_horizontal,
                            'fov_vertical': fov_vertical
                        }

        return cameras


    def cam_to_world_transform(self, extriniscs):
        # Extract rotation and translation data from the dictionary
        qw, qx, qy, qz = extriniscs['qw'], extriniscs['qx'], extriniscs['qy'], extriniscs['qz']
        tx, ty, tz = extriniscs['tx'], extriniscs['ty'], extriniscs['tz']
        
        # Convert quaternion to a rotation matrix
        rot_matrix = R.from_quat([qw, qx, qy, qz]).as_matrix()

        # Invert the rotation matrix (transpose, because rotation matrices are orthogonal)
        rot_matrix_inv = rot_matrix.T

        # Compute the camera position in world coordinates (applying -R^T * t)
        translation = np.array([tx, ty, tz])
        cam_position = -np.dot(rot_matrix_inv, translation)

        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix_inv
        transform_matrix[:3, 3] = cam_position

        return torch.from_numpy(transform_matrix)

    def compute_T(self, extriniscs_list, use_norm = True):
        target_cam2world = self.cam_to_world_transform(extriniscs_list[0])
        target_cam2world = target_cam2world.unsqueeze(0)

        cond_cam2world = torch.stack([self.cam_to_world_transform(extriniscs) for extriniscs in extriniscs_list], dim=0)
        batch_size = cond_cam2world.shape[0]
        relative_target_transformation = torch.linalg.inv(cond_cam2world) @ target_cam2world
        if use_norm:
            relative_target_transformation = self.scale_translation_norms(relative_target_transformation)
        assert relative_target_transformation.shape == (batch_size, 4, 4)
        return relative_target_transformation
    

    def scale_translation_norms(self, cond_cam2world_matrices):
        """
        Scale the translation vectors of cond_cam2world matrices based on the maximum norm,
        but return the full transformation matrix with scaled translations.

        Parameters:
        - cond_cam2world_matrices (torch.Tensor): A batch of 4x4 transformation matrices.
        
        Returns:
        - torch.Tensor: Transformation matrices with scaled translations.
        """
        # 提取平移向量（假设平移向量在每个矩阵的第四列前三行）
        translations = cond_cam2world_matrices[:, :3, 3]

        # 计算平移向量的模长
        norms = torch.linalg.norm(translations, dim=1)
        # 获取最大模长，并避免除以非常小的数
        max_norm = torch.max(norms)
        cond_cam2world_matrices[:, :3, -1] /= torch.clip(max_norm, min=1e-2, max=None)

        return cond_cam2world_matrices
    


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents

def vae_latent_to_tensor(latent, vae):
    video_length = latent.shape[1]
    latent = rearrange(latent, "b f c h w -> (b f) c h w") / vae.config.scaling_factor
    t = vae.decode(latent, num_frames=video_length).sample
    t = rearrange(t, "(b f) c h w -> b f c h w", f=video_length)
    return t

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=26,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        help="The directory where has the sparse view reconstruction results.",
    )
    parser.add_argument(
        "--ref_folders",
        type=str,
        help="The directory where has ground truth for the sparse view reconstruction results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    print("args local rank: ", args.local_rank, accelerator.process_index)
    torch.cuda.set_device(args.local_rank)

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    
    ori_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
        ignore_mismatched_sizes=True,
    )

    unet = UNetSpatioTemporalFlowConditionModel()
    ori_unet_dict = ori_unet.state_dict()
    unet_dict = unet.state_dict()

    filtered_unet_dict = {}
    untransferred_keys = []

    for key in unet_dict:
        if key in ori_unet_dict and ori_unet_dict[key].size() == unet_dict[key].size():
            filtered_unet_dict[key] = ori_unet_dict[key]
        else:
            untransferred_keys.append(key)
    
    unet_dict.update(filtered_unet_dict)
    unet.load_state_dict(unet_dict)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalFlowConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalFlowConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalFlowConditionModel.from_pretrained(
                    input_dir, subfolder="unet")

                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # cc_projection.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    unet.requires_grad_(True)
    # cc_projection.requires_grad_(True)
    parameters_list = []

    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.

    for name, para in unet.named_parameters():
        # print("Unet name: ", name)
        if 'temporal_transformer_block' in name:
            parameters_list.append(para)
            para.requires_grad = True
        elif 'pose_embedding' in name:
            parameters_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = DummyDataset(base_folder=args.base_folder, ref_folders=args.ref_folders, \
                                 width=args.width, height=args.height, sample_frames=args.num_frames)
    print("training dataset length: ", len(train_dataset))
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings
        

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        poses,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.module.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(poses.device)
        add_time_ids = add_time_ids.repeat(poses.shape[0], 1)
        add_time_ids = torch.cat([add_time_ids, poses], dim=-1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        # cc_projection.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate([unet]):
                # first, convert images to latent space.
                pixel_values = batch["pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )

                conditional_pixel_values = batch["condition_pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )

                conditional_T = batch["condition_T"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                conditional_T = conditional_T.squeeze(0)

                # we hope the first frame and last frame should be clear
                conditional_pixel_values[:, 0:1, :, :, :] = pixel_values[:, 0:1, :, :, :]
                conditional_pixel_values[:, -1, :, :, :] = pixel_values[:, -1, :, :, :]

                conditional_pixel_values_clone = conditional_pixel_values.clone().squeeze(0)
                pixel_values_clone = pixel_values.clone().squeeze(0)

                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                frames_num = latents.shape[1]
                # print("----------latents shape: ", latents.shape)

                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1

                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae) / vae.config.scaling_factor

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                # print("-----------sigmas: ", sigmas)
                noisy_latents = latents + noise * sigmas
                # c_noise =  0.25 * torch.log(sigmas)
                c_in = 1 / ((sigmas**2 + 1) ** 0.5)
                
                inp_noisy_latents = noisy_latents * c_in
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                # inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                # encoder_hidden_states = encode_image(
                #     pixel_values[:, 0, :, :, :].float())
                # print("original pixel values shape: ", pixel_values.shape)
                conditional_pixel_values = conditional_pixel_values.squeeze(0)
                encoder_hidden_states = encode_image(
                    conditional_pixel_values.float())
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                # print("after changing original pixel values shape: ", encoder_hidden_states.shape)

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    conditional_T,
                    encoder_hidden_states.dtype,
                    bsz,
                )
                # added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        frames_num, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(frames_num, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    prompt_mask[0] = 0
                    prompt_mask[-1] = 0
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)
                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(frames_num, 1, 1, 1)
                    image_mask[0] = 1
                    image_mask[-1] = 1 
                    # Final image conditioning.
                    # print("---------------img mask shape: ", image_mask.shape)
                    conditional_latents = image_mask * conditional_latents
                    
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)
                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents
                model_pred = unet(
                    inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample
                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                loss = torch.mean(
                    (weighing.float() * torch.abs(denoised_latents.float() -
                     target.float())).reshape(target.shape[0], -1),
                    dim=1,
                )

                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0)
                        or (global_step == 1)
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = StableFlowVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(
                                image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            # cc_projection=accelerator.unwrap_model(cc_projection),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):
                            for val_img_idx in range(args.num_validation_images):
                                num_frames = args.num_frames
                                # print("inference frames: ", num_frames)
                                conditional_pixel_values = (conditional_pixel_values_clone + 1.0) / 2.0
                                video_frames = pipeline(
                                    conditional_pixel_values,
                                    poses = conditional_T,
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    decode_chunk_size=8,
                                    motion_bucket_id=127,
                                    fps=7,
                                    noise_aug_strength=0.02,
                                    # generator=generator,
                                ).frames[0]

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.mp4",
                                )

                                out_cond_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_cond_img_{val_img_idx}.mp4",
                                )

                                out_gt_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_gt_img_{val_img_idx}.mp4",
                                )

                                out_vae_gt_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_vae_gt_img_{val_img_idx}.mp4",
                                )

                                for i in range(num_frames):
                                    img = video_frames[i]
                                    video_frames[i] = np.array(img)
                                    # print("img shape: ", video_frames[i].shape)
                                export_to_gif(video_frames, out_file, 8)
                                
                                transform = transforms.ToPILImage()
                                for i in range(len(conditional_pixel_values_clone)):
                                    img = conditional_pixel_values_clone[i]
                                    img = transform((img + 1.0) / 2)
                                    video_frames[i] = np.array(img)
                                    # print("cond img shape: ", video_frames[i].shape)
                                export_to_gif(video_frames, out_cond_file, 8)

                                gt_latents = tensor_to_vae_latent(pixel_values_clone[None,:, :, :, :], vae)
                                gt_tensor = vae_latent_to_tensor(gt_latents, vae).squeeze(0)

                                for i in range(len(gt_tensor)):
                                    img = gt_tensor[i]
                                    img = transform((img + 1.0) / 2)
                                    video_frames[i] = np.array(img)
                                export_to_gif(video_frames, out_vae_gt_file, 8)

                                for i in range(len(pixel_values_clone)):
                                    img = pixel_values_clone[i]
                                    img = transform((img + 1.0) / 2)
                                    video_frames[i] = np.array(img)
                                    # print("cond img shape: ", video_frames[i].shape)
                                export_to_gif(video_frames, out_gt_file, 8)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableFlowVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            # cc_projection=accelerator.unwrap_model(cc_projection),
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
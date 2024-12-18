import torch
import os
import numpy as np
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel


from torch.utils.data import Dataset
from train_svd import _gaussian_blur2d, tensor_to_vae_latent, rand_log_normal
from pathlib import Path
from einops import rearrange

import argparse

class DummyInferenceDataset(Dataset):
    def __init__(self, num_samples=100000, width=1024, height=576, sample_frames=25, root_path='', \
        scene_txt='', ref_path='/scratch/xi9/DATASET/MIPNERF-test-3/Ref', num_views=3, max_ratio=2, \
        dataset='DL3DV'):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        self.base_folder = root_path
        self.scenes = []
        with open(scene_txt, 'r') as file:
            for line in file:
                directory_path = line.strip()
                self.scenes.append(directory_path)
        self.LR_scenes = self.scenes
        self.ref_path = ref_path
        self.dataset = dataset

        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.samples = []
        self.num_views = num_views
        self.dataset = dataset

        # scenes = [os.path.join(self.base_folder, f) for f in self.folders]
        for scene in  self.scenes:
            scene_path = os.path.join(self.base_folder, scene, 'hr', f'test_{self.num_views}/ours_30000/renders')
            frames = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path)])
            if self.dataset == 'DL3DV':
                ref_indices = [index for index, value in enumerate(frames) if 'frame' in value]
            elif self.dataset == 'MipNeRF':
                ref_indices = [index for index, value in enumerate(frames) if 'DSC' in value]
            elif self.dataset == 'LLFF':
                ref_indices = [index for index, value in enumerate(frames) if len(os.path.basename(value)) > 9]
            else:
                raise ValueError(f"Do not support {self.dataset} dataset")
            
            # for start, end in zip(ref_indices, ref_indices[1:] + [None]):
            #     if end is not None:
            #         cur_list = frames[start:end]
            #         samples = self.create_interleaved_sublists(cur_list, sample_frames - 1)
            #         samples = [sample_ + [frames[end]] for sample_ in samples]
            #         self.samples.extend(samples)

            for start, end in zip(ref_indices, ref_indices[1:] + [None]):
                if end is not None:
                    cur_list = frames[start:(end+1)]
                    samples = self.uniform_sample_with_fixed_count(cur_list, self.sample_frames)
                    self.samples.append(samples)
    
    def uniform_sample_with_fixed_count(self, lst, count):
        length = len(lst)
        if length < count:
            raise ValueError("The list is too short to sample the desired number of elements.")

        sampled_list = [lst[0]]
        remaining_count = count - 2
        sublist = lst[1:-1]
        interval = len(sublist) // remaining_count
        start = int((len(lst) - interval * 23) / 2)
        
        for i in range(remaining_count):
            index = (start + i * interval) % len(sublist)
            sampled_list.append(sublist[index])

        sampled_list.append(lst[-1])
        print(len(sampled_list))
        return sampled_list
    
    def create_interleaved_sublists(self, data, num_groups=5):
        first_element = data[0]
        remaining_data = data[1:]

        group_size = len(remaining_data) // (num_groups - 1)
        extras = len(remaining_data) % (num_groups - 1)

        groups = []
        start_index = 0
        for i in range(num_groups - 1):
            end_index = start_index + group_size + (1 if i < extras else 0)
            groups.append(remaining_data[start_index:end_index])
            start_index = end_index

        result = []
        max_len = max(len(group) for group in groups)
        for i in range(max_len):
            sublist = [first_element]
            for group in groups:
                if i < len(group):
                    sublist.append(group[i])
                else:
                    sublist.append(data[-1]) 

            result.append(sublist)

        return result                


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths = self.samples[idx]
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        ref_pixel_values = torch.empty((2, self.channels, self.height, self.width))

        for i, frame_path in enumerate(image_paths):
            # with Image.open(frame_path) as img:
            img = self.open_image(frame_path)
            # Resize the image and convert it to a tensor
            img_resized = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

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
        
        for i, frame_path in enumerate(image_paths):
            img = self.open_image(frame_path.replace("/hr/", f"/{self.num_views}/"))
            # with Image.open(frame_path.replace("/hr/", f"/{self.num_views}/")) as img:

            # Resize the image and convert it to a tensor
            img_resized = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

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
        
        ref_image_path = [image_paths[0], image_paths[-1]]
        ref_image_path = [self.modify_filename(img_path) for img_path in ref_image_path]
        # ref_image_path = [image_paths[0].replace("", "Ref"), image_paths[-1].replace("HR", "Ref")]
        for i, img_path in enumerate(ref_image_path):
            img = self.center_crop_to_shorter_axis(img_path)
            img_resized = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            # Rearrange channels if necessary
            if self.channels == 3:
                img_normalized = img_normalized.permute(
                    2, 0, 1)  # For RGB images
            elif self.channels == 1:
                img_normalized = img_normalized.mean(
                    dim=2, keepdim=True)  # For grayscale images
            ref_pixel_values[i] = img_normalized
        
        #Should load the real image

        condition_pixel_values[0, :, :, :] = ref_pixel_values[0, :, :, :]
        condition_pixel_values[-1, :, :, :] = ref_pixel_values[-1, :, :, :]

        return {'pixel_values': pixel_values, "condition_pixel_values":condition_pixel_values, "image_paths": image_paths}
    
    def modify_filename(self, path):
        # Split the path into directory and file name
        directory, filename = os.path.split(path)
        p = Path(directory)
        scene = p.parts[-5]

        
        # Create the new filename from the 'frame' substring onwards
        new_filename = filename[5:]
        
        # Join the directory and the new filename to get the full path
        # print("ref path is: ", self.ref_path, scene, new_filename)
        new_path = os.path.join(self.ref_path, scene, new_filename)
       
        if self.dataset == 'LLFF':
            new_path = os.path.join(self.ref_path, scene, 'images', new_filename)
        elif self.dataset == "MipNeRF":
            new_path = os.path.join(self.ref_path, scene, 'images', new_filename)
            # new_path = new_path.replace('png', 'JPG')
        print("ref path is: ", new_path)
        
        return new_path

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


import argparse   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--root_path", type=str, help="Root path of dataset")
    parser.add_argument("--ref_path", type=str, help="Root path of ref dataset")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--num_views", type=str, help="num of reference views")
    parser.add_argument("--datatype", type=str, help="num of reference views")
    args = parser.parse_args()

    pretrained_model_name_or_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
    revision = None
    weight_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder", revision=revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16",
        unet=unet, image_encoder=image_encoder, vae=vae, revision=revision,
    )


    width=512
    height=512
    num_frames=25
    train_dataset = DummyInferenceDataset(width=width, height=height, sample_frames=num_frames, root_path=args.root_path, \
        ref_path=args.ref_path, scene_txt=args.scene, num_views=int(args.num_views), dataset=args.datatype)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
    )

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


    # Load the conditioning image
    def encode_image(pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
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
        # print("training image encoder input shape: ", pixel_values.shape)
        pixel_values = pixel_values.to(
            device=device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings
            

    def _get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
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
            add_time_ids = add_time_ids.repeat(batch_size, 1)
            return add_time_ids

    # pipeline = pipeline.to(device)

    outdir = args.output_path
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    input_path = os.path.join(outpath, "inputs")
    os.makedirs(input_path, exist_ok=True)
    gt_path = os.path.join(outpath, "gts")
    os.makedirs(gt_path, exist_ok=True)
    latent_path = os.path.join(outpath, "latents")
    os.makedirs(latent_path, exist_ok=True)
    for step, batch in enumerate(train_dataloader):
        # first, convert images to latent space.
        pixel_values = batch["pixel_values"].to(weight_dtype).to(
            device, non_blocking=True
        )

        conditional_pixel_values = batch["condition_pixel_values"].to(weight_dtype).to(
            device, non_blocking=True
        )

        image_paths = batch["image_paths"]

        # we hope the first frame and last frame should be clear

        latents = tensor_to_vae_latent(pixel_values, vae)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
        noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
        # cond_sigmas = cond_sigmas[:, None, None, None, None]
        conditional_pixel_values_clone = conditional_pixel_values.clone().squeeze(0)
        pixel_values_clone = pixel_values.clone().squeeze(0)
        conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae) / vae.config.scaling_factor

        # Sample a random timestep for each image
        # P_mean=0.7 P_std=1.6
        sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = sigmas[:, None, None, None, None]
        noisy_latents = latents
        # noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor(
            [0.25 * sigma.log() for sigma in sigmas]).to(device)

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        conditional_pixel_values = conditional_pixel_values.squeeze(0)
        conditional_pixel_values = (conditional_pixel_values_clone + 1.0) / 2.0
        
        video_frames = pipeline(
            conditional_pixel_values,
            height=height,
            width=width,
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
        ).frames[0]
        assert len(image_paths) == len(video_frames)

        for i in range(num_frames):
            img = video_frames[i]
            video_frames[i] = np.array(img)
            img_path = Path(image_paths[i][0])
            scene =img_path.parts[-6]
            img_name = img_path.parts[-1]
            scene_dir = os.path.join(sample_path, scene)
            print("scene_dir: ", scene_dir, scene)
            if not os.path.isdir(scene_dir):
                os.makedirs(scene_dir, exist_ok=True) 
            Image.fromarray(video_frames[i].astype(np.uint8)).save(
                                        os.path.join(scene_dir, img_name))

            scene_dir = os.path.join(input_path, scene)
            if not os.path.isdir(scene_dir):
                os.makedirs(scene_dir, exist_ok=True)

            cond_img = 255 * conditional_pixel_values[i, :, :, :]
            cond_img = rearrange(cond_img.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(cond_img.astype(np.uint8)).save(
                                        os.path.join(scene_dir, img_name))
            
            scene_dir = os.path.join(gt_path, scene)
            if not os.path.isdir(scene_dir):
                os.makedirs(scene_dir, exist_ok=True)

            gt_img = 255 * pixel_values[i, :, :, :]
            gt_img = rearrange(gt_img.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(gt_img.astype(np.uint8)).save(
                                        os.path.join(scene_dir, img_name))
            

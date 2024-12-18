import torchvision.transforms as transforms
import torch.nn as nn
import random
import os
import copy
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class VideoDataset(Dataset):
    def __init__(self, width=1024, height=576, base_folder='/scratch/xi9/DATASET/DL3DV-960P-2K-Randominit', \
                 ref_folders='/scratch/xi9/Large-DATASET/DL3DV-10K/2K', sample_frames=26, max_ratio=3):
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
        # print("base path: ", os.path.join(self.base_folder, self.scenes[0], 'lr'))
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
                        self.samples[scene][view_num].append(sample)
                        self.length = self.length + 1
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
            img = self.center_crop(frame_path, self.width, self.height)
            img_tensor = torch.from_numpy(np.array(img)).float()

            img_normalized = img_tensor / 127.5 - 1

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
            img = self.center_crop(gt_frame_path, self.width, self.height)
            img_tensor = torch.from_numpy(np.array(img)).float()
            img_normalized = img_tensor / 127.5 - 1

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
        crop_size = min(width, height)
        
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
    

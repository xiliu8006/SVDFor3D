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

class VideoInferenceDataset(Dataset):
    def __init__(self, width=1024, height=576, base_folder='/scratch/xi9/DATASET/DL3DV-960P-2K-Randominit', \
                 ref_folders='/scratch/xi9/Large-DATASET/DL3DV-10K/2K', sample_frames=26, max_ratio=3, render_all=True):
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
        self.samples = []
        self.random_samples = {}
        # print("base path: ", os.path.join(self.base_folder, self.scenes[0], 'lr'))
        self.view_num = '9'
        self.length = 0
        self.valid_scenes = []

        for scene in self.scenes:
            # print("scene: ", scene)
            scene_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'train_'+ self.view_num)
            if not os.path.exists(scene_path):
                continue
            self.valid_scenes.append(scene)
            frames = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path)])
            if len(frames) < sample_frames:
                continue
            ref_indices = [index for index, value in enumerate(frames) if '_ref' in value]
            if render_all:
                for start, end in zip(ref_indices, ref_indices[1:] + [None]):
                    if end is not None:
                        cur_list = frames[start:end]
                        samples = self.create_interleaved_sublists(cur_list, sample_frames - 1)
                        samples = [sample_ + [frames[end]] for sample_ in samples]
                        self.samples.extend(samples)
            else:
                for start, end in zip(ref_indices, ref_indices[1:] + [None]):
                    if end is not None:
                        if end - start != sample_frames:
                            if start + sample_frames > len(frames):
                                sample = frames[-sample_frames:]
                            else:
                                if end-start > sample_frames:
                                    sample = frames[start:(end+1)]
                                    sample = self.uniform_sample_with_fixed_count(sample, sample_frames)
                                else:
                                    sample = frames[start: (start + sample_frames)]
                        else:
                            sample = frames[start:(end+1)]
                    if len(sample) != sample_frames:
                        print("Error len: ", len(sample), sample_frames)
                    self.samples.append(sample)

        self.scenes = self.valid_scenes
        self.colmap = {}
        self.camera = {}
        self.inferenced_image = {}
        for scene in self.scenes:
            images_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'sparse/0_render/images.txt')
            camera_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'sparse/0_render/cameras.txt')
            self.colmap[scene] = self.read_colmap_images(images_path)
            self.camera[scene] = self.load_cameras_and_calculate_fov(camera_path)
        # print("samples and scenes len: ", len(self.samples), len(self.scenes), self.scenes)

    def __len__(self):
        return len(self.samples)

    def copy_colmap_files(self, save_dir):
        for scene in self.scenes:
            images_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'sparse/0_render/images.txt')
            camera_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'sparse/0_render/cameras.txt')
            pc_path = os.path.join(self.base_folder, scene, 'lr', self.view_num, 'input.ply')
            colmap_dir = os.path.join(save_dir, scene, self.view_num, 'sparse/0')
            if not os.path.exists(colmap_dir):
                os.makedirs(colmap_dir)
            new_images_path = os.path.join(colmap_dir, 'images.txt')
            new_camera_path = os.path.join(colmap_dir, 'cameras.txt')

            with open(camera_path, 'r') as file:
                camera_lines = file.readlines()
                modified_camera_lines = []
                new_parts = []
                for line in camera_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():  # 只修改有效的相机定义行
                        new_parts = copy.deepcopy(parts)
                        new_parts[0] = str(2)
                        # new_parts[2] = str(self.width * 4)
                        # new_parts[3] = str(self.height * 4)
                        new_parts[2] = str(self.width * 8)
                        new_parts[3] = str(self.height * 8)
                    print("modified part: ", parts)
                    modified_camera_lines.append(" ".join(parts) + "\n")
                    modified_camera_lines.append(" ".join(new_parts) + "\n")

            with open(new_camera_path, 'w') as file:
                file.writelines(modified_camera_lines)
            
            modified_lines = []
            with open(images_path, 'r') as file:
                for line in file:
                    if line.strip() and not line.startswith('#'):  # Check if the line is not empty or a comment
                        parts = line.strip().split()
                        if 'ref' not in parts[-1]:  # Check if 'ref' is not in the NAME
                            parts[8] = '2'  # Set CAMERA_ID to 2
                        modified_lines.append(' '.join(parts))
                    else:
                        modified_lines.append(line.strip())  # Preserve empty lines and comments as they are
            
            with open(new_images_path, 'w') as file:
                for line in modified_lines:
                    file.write(line + '\n')

            new_pc_path = os.path.join(colmap_dir, 'points3D.ply')
            # shutil.copy2(images_path, new_images_path)
            # shutil.copy2(camera_path, new_camera_path)
            shutil.copy2(pc_path, new_pc_path)

            # # 定义图片目录路径
            # images_dir = os.path.join(save_dir, scene, self.view_num, 'images')
            # if not os.path.exists(images_dir):
            #     os.makedirs(images_dir)

            # # 移动所有的 .png 文件到图片目录
            # png_files = glob.glob(os.path.join(save_dir, scene, self.view_num, '*.png'))
            # for png_file in png_files:
            #     shutil.move(png_file, images_dir)

            
    def pick_n_elements_randomly(self, lst, n):
        if n > len(lst) or n < 2:
            raise ValueError("Number of elements to pick must be between 2 and the length of the list")
        
        # Ensure first and last elements are included
        first_element = lst[0]
        last_element = lst[-1]
        
        # If only two elements are needed, return them directly
        if n == 2:
            return [first_element, last_element]
        
        # Pick n-2 random elements from the middle of the list, excluding the first and last
        middle_elements = random.sample(lst[1:-1], n-2)
        
        # Combine the first and last elements with the randomly picked middle elements
        selected_elements = [first_element] + middle_elements + [last_element]
        
        return selected_elements
    
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

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        selected_frames = self.samples[idx]
        
        scene_path = selected_frames[0]
        path_parts = scene_path.split(os.sep)
        chosen_scene = path_parts[-5]
        # print("path_parts: ", path_parts, self.colmap)

        # Initialize a tensor to store the pixel values
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        poses = torch.empty((self.sample_frames, 19))
        # Load and process each frame
        extriniscs_list = []
        for i, frame_path in enumerate(selected_frames):
            frame_name = os.path.basename(frame_path)
            # assert False, f"self camera is {chosen_scene} {frame_name} {self.colmap[chosen_scene]}"
            extriniscs = self.colmap[chosen_scene][frame_name]
            extriniscs_list.append(extriniscs)
        
        poses = self.compute_T(extriniscs_list)
        poses = poses.reshape((self.sample_frames, 16))
        # assert False, f"self camera is {self.camera[chosen_scene]}"
        # print("----------------self camera is: ", self.camera[chosen_scene])
        fov_deg = self.camera[chosen_scene]['1']['fov_horizontal']
        fov_rad = torch.tensor(fov_deg * np.pi / 180)
        fov_enc = torch.stack(
            [fov_rad, torch.sin(fov_rad), torch.cos(fov_rad)], axis=-1
        )

        fov_enc = fov_enc.repeat(self.sample_frames, 1)
        T = torch.cat([poses, fov_enc], axis = 1)

        for i, frame_path in enumerate(selected_frames):
            frame_name = os.path.basename(frame_path)
            frame_path = os.path.join(self.base_folder, chosen_scene, 'lr', self.view_num, f"train_{self.view_num}", frame_name)
            
            # img = self.center_crop_to_shorter_axis(frame_path)
            img = self.center_crop(frame_path, self.width, self.height)
            # img = img.resize((self.width, self.height))
            img_tensor = torch.from_numpy(np.array(img)).float()

            if i == 0:
                self.height = img.height
                self.width = img.width
                pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
                condition_pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

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
        
        for i, frame_path in enumerate(selected_frames):
            frame_name = os.path.basename(frame_path)
            ref_frame_name = frame_name.replace('_ref', '')
            gt_frame_path = os.path.join(self.ref_folders, chosen_scene, 'images_4', ref_frame_name)

            # img = self.center_crop_to_shorter_axis(gt_frame_path)
            # img_resized = img.resize((self.width, self.height))
            # img_tensor = torch.from_numpy(np.array(img_resized)).float()

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

        return {'pixel_values': pixel_values, "condition_pixel_values":condition_pixel_values, "condition_T": T, "image_paths": selected_frames}
    
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
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', 'JPG']  # 添加你认为可能的图片格式
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
                    image_id = parts[0]
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = parts[8]
                    image_name = parts[9]

                    images[image_name.replace('JPG', 'png')] = {
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
                    
                    # Skip the next line which is expected to be empty or contain point data
                    i += 1
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
                    # print("parts: ", parts, len(parts))
                    if len(parts) >= 8:  # Check for at least enough parts for a camera entry
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
    

import torch
import argparse
from torch.utils.data import Dataset

from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import os
from itertools import chain

class EvalDataset(Dataset):
    def __init__(self, width=1024, height=576, sample_frames=26, max_ratio=3, method = 'nodropout_camP'):
        self.base_folder = f'/scratch/xi9/DATASET/DL3DV-960P-Benchmark-SVD/{method}/samples'
        self.ref_folders = '/scratch/xi9/Large-DATASET/DL3DV-10K/1K'
        self.base_scenes = set(os.listdir(self.base_folder))
        self.ref_scenes = set(os.listdir(self.ref_folders))
        self.scenes = list(self.base_scenes.intersection(self.ref_scenes))

    def __len__(self):
        return len(self.scenes)
    
    def get_scene(self, idx):
        return self.scenes[idx]
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        lr_frames = {}
        gt_frames = {}
        for view_num in os.listdir(os.path.join(self.base_folder, scene)):
            scene_path = os.path.join(self.base_folder, scene, view_num, 'images')
            cur_lr_frames = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if '_ref' not in img])
            cur_gt_frames = []
            for i, frame_path in enumerate(cur_lr_frames):
                ref_frame_name = os.path.basename(frame_path).replace('_ref', '')
                gt_frame_path = os.path.join(self.ref_folders, scene, 'images_4', ref_frame_name)
                cur_gt_frames.append(gt_frame_path)
            assert len(cur_gt_frames) == len(cur_lr_frames), f"{len(cur_gt_frames)}, {len(cur_lr_frames)}"
            lr_frames[view_num] = cur_lr_frames
            gt_frames[view_num] = cur_gt_frames
            
        return{'lr_frames': lr_frames, 'gt_frames': gt_frames}

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_image_as_tensor(image_path):
    """Load an image and convert it to a PyTorch tensor with values in [0, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # image = Image.open(image_path)
    image = center_crop(image_path, 960, 512)
    return transform(image)

def center_crop(image_path, target_width, target_height):
        # Open the image file
        img = open_image(image_path)
        
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

def open_image(file_name):
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

def process_frame(view_num, scene, lr_path, gt_path):
    lr_tensor = load_image_as_tensor(lr_path[0])
    gt_tensor = load_image_as_tensor(gt_path[0])
    return psnr(lr_tensor, gt_tensor).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="Scene name")
    args = parser.parse_args()

    eval_dataset = EvalDataset(method=args.method)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=4,
    )

    psnr_all = {}
    for step, batch in enumerate(eval_dataloader):
        lr_frames = batch['lr_frames']
        gt_frames = batch['gt_frames']
        for view_num in lr_frames.keys():
            # print("current view: ", view_num)
            scene = eval_dataset.get_scene(step)
            print("current view: ", view_num, scene, step)
            # Initialize the dictionary for new view_num and scene
            if view_num not in psnr_all:
                psnr_all[view_num] = {}
            if scene not in psnr_all[view_num]:
                psnr_all[view_num][scene] = []
            
            # Check if number of frames match in lr and gt
            assert len(lr_frames[view_num]) == len(gt_frames[view_num]), f"Frame length mismatch in view {view_num}: {len(lr_frames[view_num])} vs {len(gt_frames[view_num])}"
            
            # Load all frames for this batch
            lr_tensors = [load_image_as_tensor(lr_path[0]) for lr_path in lr_frames[view_num]]
            gt_tensors = [load_image_as_tensor(gt_path[0]) for gt_path in gt_frames[view_num]]
            
            # Stack all frames into a single batch tensor for processing
            lr_tensors = torch.stack(lr_tensors)
            gt_tensors = torch.stack(gt_tensors)

            # Calculate PSNR for the entire batch
            psnrs = psnr(lr_tensors, gt_tensors)
            print(f'scene {scene} view {view_num} with {len(psnrs)} images, psnrs: {psnrs.mean()}')
            # Store results
            print(f'Debug: PSNR values: {psnrs.tolist()}')  # Check if this is a flat list
            print(f'scene {scene} view {view_num} psnr is: {psnrs.mean()}')
            psnr_all[view_num][scene] = (psnrs.tolist())

    
    view_psnr = []
    output_lines = []

    for view_num, scenes in psnr_all.items():
        output_lines.append(f"View {view_num} results:\n")
        scene_avg_psnr = []
        for scene, psnrs in scenes.items():
            flattened_psnrs = list(chain.from_iterable(psnrs))
            avg_psnr = sum(flattened_psnrs) / len(flattened_psnrs)
            scene_avg_psnr.append(avg_psnr)
            output_lines.append(f"  Scene {scene} with {len(psnrs)} images average PSNR: {avg_psnr:.2f}\n")
        view_average = sum(scene_avg_psnr) / len(scene_avg_psnr)
        view_psnr.append(view_average)
        output_lines.append(f"Average PSNR for View {view_num}: {view_average:.2f}\n")
    total_average_psnr = sum(view_psnr) / len(view_psnr)
    output_lines.append(f"Total average PSNR across all views: {total_average_psnr:.2f}\n")
    
    file_path =  f'/scratch/xi9/DATASET/DL3DV-960P-Benchmark-SVD/{args.method}/psnr_results.txt'
    with open(file_path, 'w') as f:
        f.writelines(output_lines)

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import json
import cv2
import torch.nn as nn
from net.pspnet import PSPNet
import logging
import subprocess
import os
from rembg import remove
import sys
import os

from lightweight_human_pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
from lightweight_human_pose_estimation.modules.load_state import load_state
from lightweight_human_pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
from lightweight_human_pose_estimation.modules.pose import Pose
from lightweight_human_pose_estimation.val import normalize, pad_width
import torchvision.transforms as T
import json
from lightweight_human_pose_estimation.modules.pose import Pose, track_poses
import matplotlib.pyplot as plt
import numpy as np

class CPDataset(data.Dataset):
    """
    Dataset for CP-VTON.
    """

    def __init__(self, opt, person_img, cloth_img):
        super().__init__()

        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)

        self.pose_net = PoseEstimationWithMobileNet()
        checkpoint_path = "checkpoints/checkpoint_iter_370000.pth"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        load_state(self.pose_net, checkpoint)
        self.pose_net = self.pose_net.cuda() if torch.cuda.is_available() else self.pose_net
        self.pose_net.eval()

        self.transformRGB = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transformL = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.img_names = person_img
        self.cloth_names = cloth_img

    def name(self):
        return "CPDataset"
    

    import numpy as np

# Helper function to recursively convert ndarray to list
    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray_to_list(elem) for elem in obj]
        else:
            return obj



    def extract_pose_keypoints(self, pil_image, track=True, smooth=True, save_path='output_pose_result.jpg'):

        image = np.array(pil_image.convert("RGB"))
        orig_img = image.copy()

        height, width, _ = image.shape
        stride = 8
        upsample_ratio = 4
        height_size = 256
        scale = height_size / height
        scaled_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, np.array([128, 128, 128], np.float32), np.float32(1 / 256))

        min_dims = [height_size, max(scaled_img.shape[1], height_size)]
        padded_img, pad = pad_width(scaled_img, stride, 0, min_dims)
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()


        with torch.no_grad():
            stages_output = self.pose_net(tensor_img)
            
        stage2_heatmaps = stages_output[-2]
        stage2_pafs = stages_output[-1]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
        pafs = np.transpose(stage2_pafs.squeeze().cpu().numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        num_keypoints = Pose.num_kpts
        total_keypoints_num = 0
        all_keypoints_by_type = []

        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        pose_dicts = []
        for pose in current_poses:
            pose_dicts.append({
        "keypoints": pose.keypoints.tolist() if isinstance(pose.keypoints, np.ndarray) else pose.keypoints,
        "bbox": list(pose.bbox) if isinstance(pose.bbox, (list, tuple, np.ndarray)) else pose.bbox
    })
        return pose_dicts

    def create_cloth_mask(self, cloth_image):
        """ Create a binary mask for the given cloth image. """
        if isinstance(cloth_image, Image.Image):
            cloth_image = remove(cloth_image)
            cloth = np.array(cloth_image)
        else:
            cloth = cloth_image  # Assume it's already a NumPy array

        # Extract the alpha channel (transparency mask)
        if cloth.shape[-1] == 4:  # Check if there is an alpha channel
            alpha_channel = cloth[:, :, 3]
            mask = alpha_channel > 10  # Binary mask of non-transparent areas
            cloth_rgb = cloth[:, :, :3]  # Extract RGB channels
            # Set transparent areas to white
            cloth_rgb[mask == 0] = [255, 255, 255]
        
        else:
            cloth_rgb = cloth  # No alpha channel, use as-is
            
        # Fallback if the image does not have an alpha channel
        gray = cv2.cvtColor(cloth, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
        # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Crop the cloth image and mask
            cloth_cropped = cloth[y:y+h, x:x+w]
            mask_cropped = mask[y:y+h, x:x+w]

        # Resize to 192x256
            cloth_resized = cv2.resize(cloth_cropped, (192, 256), interpolation=cv2.INTER_CUBIC)
            mask_resized = cv2.resize(mask_cropped, (192, 256), interpolation=cv2.INTER_NEAREST)
        else:
        # If no contour is found, return a blank image and mask
            cloth_resized = np.ones((256, 192, 3), dtype=np.uint8) * 255  # White background
            mask_resized = np.zeros((256, 192), dtype=np.uint8)

        return cloth_resized, mask_resized

    
    def create_cloth_mask_from_person(self, person_image):
        """ Extract the cloth from the person using parsing results and set non-cloth areas to white. """
    
    # Run parsing segmentation
        parse_array = self.run_inference(person_image)

    # Extract the upper cloth mask (label 5)
        parse_cloth =(parse_array == 5).astype(np.float32)*255  + \
                     (parse_array == 6).astype(np.float32)*255 + \
                     (parse_array == 7).astype(np.float32) *255 

    # Smooth the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        parse_cloth = cv2.morphologyEx(parse_cloth, cv2.MORPH_CLOSE, kernel)

    # Convert to PIL Image
        parse_cloth_pil = Image.fromarray(parse_cloth)

    # Convert input image to NumPy array
        person_np = np.array(person_image)

    # Create a white background
        white_background = np.ones_like(person_np) * 255  # Pure white image

    # Use the mask to keep only the cloth area
        result = np.where(parse_cloth[:, :, None] == 255, person_np, white_background)
        # Convert extracted cloth to an Image object
        cloth_image = Image.fromarray(result)

    # Process the extracted cloth for further refinement
        refined_cloth, refined_mask = self.create_cloth_mask(cloth_image)

        return refined_cloth, refined_mask

    def build_network(self, snapshot, backend):
        """Loads the pre-trained model."""
        models = {
            'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
        }
        backend = backend.lower()
        net = models[backend]()
        net = nn.DataParallel(net)

        if snapshot is not None:
            net.load_state_dict(torch.load(snapshot, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            logging.info(f"Loaded model from {snapshot}")

        net = net.cuda() if torch.cuda.is_available() else net
        return net


    
    def run_inference(self, image_input):
        """Runs inference and returns the segmented image as a PIL image."""

        snapshot = './checkpoints/densenet/PSPNet_last'
        net = self.build_network(snapshot, 'densenet')
        net.eval()

        if isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = image_input  # Assume it's already a PIL Image

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img = transform(img).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        img = img.cuda() if torch.cuda.is_available() else img

        with torch.no_grad():
            pred, _ = net(img)
            pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred = np.argmax(pred, axis=2).astype(np.uint8)
            unique_classes = np.unique(pred)
            print("Unique Classes in Segmentation:", unique_classes)

        return pred

    def process_pil_image_with_openpose(self,pil_image):
        """ Process a PIL image with OpenPose and return keypoints as a NumPy array. """
# Paths to OpenPose
        OPENPOSE_BIN = "openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/openpose/bin/OpenPoseDemo.exe"
        MODEL_FOLDER = "openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d/openpose/models/"
# Input image path
        IMAGE_FOLDER = "inputs/"
# Output folder (will be used, but JSON won't be saved)
        OUTPUT_FOLDER = "output_json/"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Convert PIL image to OpenCV format
        image_cv = np.array(pil_image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenPose

    # Save the temporary image in `inputs/`
        temp_filename = os.path.join(IMAGE_FOLDER, "temp_image.jpg")
        cv2.imwrite(temp_filename, image_cv)

    # Run OpenPose
        cmd = [
        OPENPOSE_BIN,
        "--image_dir", IMAGE_FOLDER,
        "--model_folder", MODEL_FOLDER,
        "--write_json", OUTPUT_FOLDER,
        "--model_pose", "COCO",
        "--display", "0", "--render_pose", "0"
    ]
    
        subprocess.run(cmd)  # Run OpenPose silently

    # Delete the temporary image file
        os.remove(temp_filename)

    # Get JSON output file
        json_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.json')]
        if not json_files:
            print("❌ No keypoints detected!")
            return None

        json_path = os.path.join(OUTPUT_FOLDER, json_files[0])
        with open(json_path, 'r') as f:
            data = json.load(f)
        os.remove(json_path)  # Delete JSON immediately after reading

    # Extract keypoints
        people = data.get("people", [])
        if len(people) == 0:
            print("❌ No person detected in the image!")
            return None

    # Get pose keypoints
        pose_keypoints = people[0].get("pose_keypoints_2d", [])
        pose_data = np.array(pose_keypoints).reshape((-1, 3))  # shape: (18, 3)

        return pose_data

    def __getitem__(self, idx=None):
        cloth_name = self.cloth_names
        img_name = self.img_names

        cloth_np = np.array(cloth_name)
        img_np = np.array(img_name)

        cloth, cloth_mask = self.create_cloth_mask_from_person(cloth_np)

        cloth_pil = Image.fromarray(cloth)
        cloth_mask_pil = Image.fromarray(cloth_mask)


        c = self.transformRGB(cloth_pil.convert("RGB"))
        cm = torch.from_numpy((np.array(cloth_mask_pil) >= 128).astype(np.float32)).unsqueeze(0)

        im = self.transformRGB(img_name)

        im_parse = self.run_inference(img_name)
        im_parse=Image.fromarray(im_parse)

        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_headandLC =(parse_array == 2).astype(np.float32)+ \
                          (parse_array == 4).astype(np.float32)+ \
                     (parse_array == 9).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32)  + \
                     (parse_array == 13).astype(np.float32) 
        parse_cloth = (parse_array == 5).astype(np.float32)  + \
                     (parse_array == 6).astype(np.float32)   + \
                     (parse_array == 7).astype(np.float32)   
        print("parse_shape shape:", parse_shape.shape)
        print("parse_shape dtype:", parse_shape.dtype)
        print("im shape:", im.shape)
        

        parse_shape =  Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transformL(parse_shape)
        phead = torch.from_numpy(parse_headandLC)
        pcm = torch.from_numpy(parse_cloth)
        print("pcm shape:", pcm.shape)

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
   
        pose_data = self.extract_pose_keypoints(img_name)
        pose_data = self.convert_ndarray_to_list(pose_data)
        
        keypoints = pose_data[0]["keypoints"]
        if isinstance(keypoints[0], dict):
            pose_array = np.array([
        [kp['x'], kp['y']] for kp in keypoints
    ], dtype=np.float32)
        else:
            pose_array = np.array(keypoints, dtype=np.float32)

        point_num = pose_array.shape[0]  # Should be 18
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            x, y = pose_array[i]
            if x > 0 and y > 0:
        # Draw a small white dot at the keypoint location
                pose_img = np.zeros((self.fine_height, self.fine_width), dtype=np.uint8)
                pose_draw.rectangle((x - r, y - r, x + r, y + r), fill='white', outline='white')
                cv2.circle(pose_img, (int(x), int(y)), 4, 255, -1)
                pose_map[i] = torch.from_numpy(pose_img.astype(np.float32)) / 255.0

        # just for visualization
        im_pose = self.transformL(im_pose)

        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0)  # shape(1), im_h(3), pose_map(18) => [22, 256, 192]


        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transformL(im_g)
        else:
            im_g = ''

        result = {
            'c_name'      : cloth_name,     # for visualization
            'im_name'     : img_name,       # for visualization or ground truth
            'cloth'       : c,              # for input
            'cloth_mask'  : cm,             # for input
            'image'       : im,             # for visualization
            'agnostic'    : agnostic,       # for input
            'parse_cloth' : im_c,           # for ground truth
            'shape'       : shape,          # for visualization
            'headandLC'   : im_h,           # for visualization
            'pose_image'  : im_pose,        # for visualization
            'grid_image'  : im_g,           # for visualization
            'parse_image' : im_parse,       # for visualization
        }

        return result

    def __len__(self):
        return len(self.img_names)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super().__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            sampler=train_sampler
        )

        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()

        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default = False)

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()
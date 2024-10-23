import random
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class ImageSynthesizer:
    def __init__(self, imgpath, scales, margins, focal_length, ratio, roll, pitch, yaw, radius, intensity, ksize, sigX, mean, sigma, motion_blur_size, reflection):
        self.image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.image.shape[:2]
        # self.image = cv2.resize(self.image, (self.width*2, self.height*2), interpolation=cv2.INTER_NEAREST)
        # self.height, self.width = self.image.shape[:2]

        self.scales = scales
        self.margins = margins
        self.focal_length = focal_length
        self.min_ratio = ratio[0]
        self.max_ratio = ratio[1]
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        self.radius = radius
        self.intensity = intensity
        self.ksize = ksize
        self.sigX = sigX
        self.mean = mean
        self.sigma = sigma
        self.motion_blur_size = motion_blur_size
        self.reflection = reflection
        ref_idx = random.randint(0, 9)
        self.refpath = "drones/" + str(ref_idx) + ".jpg"
    
    def gen_nested_tag(self):
        s_outer = 14
        s_inner = 8
        im_out = np.zeros((self.scales[2]*s_outer,self.scales[2]*s_outer,4),dtype=np.uint8)
        num_tag = random.randint(0, 8)
        tag_location = ['outer']
        selection = ['tl', 'tl_inner', 'tr', 'tr_inner', 'br', 'br_inner', 'bl', 'bl_inner']

        if num_tag > 0:
            selected_tags = random.sample(selection, num_tag)
            for inner_tag in ['tl_inner', 'tr_inner', 'br_inner', 'bl_inner']:
                base_tag = inner_tag.split('_')[0] 
                if inner_tag in selected_tags and base_tag not in selected_tags:
                    selected_tags.append(base_tag)
                    num_tag += 1
            tag_location += selected_tags
        tag_location.sort(key=lambda x: '_inner' in x)

        IDs = random.sample(range(50), num_tag+1)
        imgs = []
        for id in IDs:
            im = cv2.imread(f"TagCustom52h12/tag52_12_{id:05d}.png", cv2.IMREAD_UNCHANGED)
            imgs.append(im)
        
        x_coords = {}
        y_coords = {}
        d = s_outer*self.scales[2]
        tag_size = d
        im_out[0:d,0:d] = cv2.resize(imgs[0], None, fx=self.scales[2], fy=self.scales[2], interpolation=cv2.INTER_NEAREST)
        corner_dict = {IDs[0]:[[0, 0], [1, 0], [1, 1], [0, 1]]}
        for i, tag_loc in enumerate(tag_location):
            if tag_loc == 'tl':
                d = s_outer*self.scales[1]
                x = self.scales[2]*(int((s_outer-s_inner)/2))+self.scales[1]*(self.margins[0])
                y = self.scales[2]*(int((s_outer-s_inner)/2))+self.scales[1]*(self.margins[1])
                x_coords[f'x_{tag_loc}'] = int(x)
                y_coords[f'y_{tag_loc}'] = int(y)
                im_out[y:y+d,x:x+d] = cv2.resize(imgs[i], None, fx=self.scales[1], fy=self.scales[1], interpolation=cv2.INTER_NEAREST)
                x_norm, y_norm, d_norm = x/tag_size, y/tag_size, d/tag_size
                corner_dict[IDs[i]] = [[x_norm, y_norm], [x_norm+d_norm, y_norm], [x_norm+d_norm, y_norm+d_norm], [x_norm, y_norm+d_norm]]
            if tag_loc == 'tr':
                d = s_outer*self.scales[1]
                x = self.scales[2]*(int(s_inner + (s_outer-s_inner)/2))-d-self.scales[1]*(self.margins[0])
                y = self.scales[2]*(int((s_outer-s_inner)/2))+self.scales[1]*(self.margins[1])
                x_coords[f'x_{tag_loc}'] = int(x)
                y_coords[f'y_{tag_loc}'] = int(y)
                im_out[y:y+d,x:x+d] = cv2.resize(imgs[i], None, fx=self.scales[1], fy=self.scales[1], interpolation=cv2.INTER_NEAREST)
                x_norm, y_norm, d_norm = x/tag_size, y/tag_size, d/tag_size
                corner_dict[IDs[i]] = [[x_norm, y_norm], [x_norm+d_norm, y_norm], [x_norm+d_norm, y_norm+d_norm], [x_norm, y_norm+d_norm]]
            if tag_loc == 'br':
                d = s_outer*self.scales[1]
                x = self.scales[2]*(int(s_inner + (s_outer-s_inner)/2))-d-self.scales[1]*(self.margins[0])
                y = self.scales[2]*(int(s_inner + (s_outer-s_inner)/2))-d-self.scales[1]*(self.margins[0])
                x_coords[f'x_{tag_loc}'] = int(x)
                y_coords[f'y_{tag_loc}'] = int(y)
                im_out[y:y+d,x:x+d] = cv2.resize(imgs[i], None, fx=self.scales[1], fy=self.scales[1], interpolation=cv2.INTER_NEAREST)
                x_norm, y_norm, d_norm = x/tag_size, y/tag_size, d/tag_size
                corner_dict[IDs[i]] = [[x_norm, y_norm], [x_norm+d_norm, y_norm], [x_norm+d_norm, y_norm+d_norm], [x_norm, y_norm+d_norm]]
            if tag_loc == 'bl':
                d = s_outer*self.scales[1]
                x = self.scales[2]*(int((s_outer-s_inner)/2))+self.scales[1]*(self.margins[1])
                y = self.scales[2]*(int(s_inner + (s_outer-s_inner)/2))-d-self.scales[1]*(self.margins[0])
                x_coords[f'x_{tag_loc}'] = int(x)
                y_coords[f'y_{tag_loc}'] = int(y)
                im_out[y:y+d,x:x+d] = cv2.resize(imgs[i], None, fx=self.scales[1], fy=self.scales[1], interpolation=cv2.INTER_NEAREST)
                x_norm, y_norm, d_norm = x/tag_size, y/tag_size, d/tag_size
                corner_dict[IDs[i]] = [[x_norm, y_norm], [x_norm+d_norm, y_norm], [x_norm+d_norm, y_norm+d_norm], [x_norm, y_norm+d_norm]]
            if tag_loc in ['tr_inner', 'tl_inner', 'bl_inner', 'br_inner']:
                x = x_coords['x_'+tag_loc.split('_')[0]]
                y = y_coords['y_'+tag_loc.split('_')[0]]
                d = s_outer*self.scales[0]
                x += self.scales[1]*(int((s_outer-s_inner)/2))+self.scales[0]*(self.margins[0])
                y += self.scales[1]*(int((s_outer-s_inner)/2))+self.scales[0]*(self.margins[1])
                im_out[y:y+d,x:x+d] = cv2.resize(imgs[i], None, fx=self.scales[0], fy=self.scales[0], interpolation=cv2.INTER_NEAREST)
                x_norm, y_norm, d_norm = x/tag_size, y/tag_size, d/tag_size
                corner_dict[IDs[i]] = [[x_norm, y_norm], [x_norm+d_norm, y_norm], [x_norm+d_norm, y_norm+d_norm], [x_norm, y_norm+d_norm]]

        bgr = im_out[:, :, :3]  # RGB channels
        alpha = im_out[:, :, 3]  # Alpha channel
        transparent_mask = alpha == 0
        gray_value = np.random.randint(150, 256)
        bgr[transparent_mask] = [gray_value, gray_value, gray_value]
        alpha[transparent_mask] = 255
        tag = np.dstack([bgr, alpha])
        tag = cv2.cvtColor(tag, cv2.COLOR_RGB2GRAY)

        return tag, corner_dict
            
    def create_tag(self):
        tag, corner_dict = self.gen_nested_tag()
        min_tag_size = int(np.min([self.height, self.width])*self.min_ratio)
        max_tag_size = int(np.min([self.height, self.width])*self.max_ratio)

        tag_size = random.randint(min_tag_size, max_tag_size)

        # Create 3D points of the tag
        tag_3d = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2,  tag_size / 2, 0],
            [-tag_size / 2,  tag_size / 2, 0]
        ], dtype=np.float32)

        cx = self.width / 2
        cy = self.height / 2
        intrinsic_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        roll_angle  = random.uniform(self.roll[0], self.roll[1])
        pitch_angle = random.uniform(self.pitch[0], self.pitch[1])
        yaw_angle   = random.uniform(self.yaw[0], self.yaw[1])
        rotation = R.from_euler('zyx', [yaw_angle, pitch_angle, roll_angle], degrees=True)
        rotation_matrix = rotation.as_matrix().astype(np.float32)
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        translation_vector = np.array([0, 0, 10], dtype=np.float32)

        # Project the 3D points onto the 2D image plane
        square_2d, _ = cv2.projectPoints(tag_3d, rotation_vector, translation_vector, intrinsic_matrix, np.zeros(5, dtype=np.float32))
        square_2d = square_2d.reshape(-1, 2)
        original_tag = np.array([
            [0, 0],
            [tag_size, 0],
            [tag_size, tag_size],
            [0, tag_size]
        ], dtype=np.float32)
        original_corners = np.array([original_tag])

        original_corners = np.array([original_tag])
        homography_matrix = cv2.getPerspectiveTransform(original_tag, square_2d)
        resized_square = cv2.resize(tag, (tag_size, tag_size), interpolation=cv2.INTER_NEAREST)
        transformed_tag = cv2.warpPerspective(resized_square, homography_matrix, (self.width, self.height))
        transformed_corners = cv2.perspectiveTransform(original_corners, homography_matrix)
        transformed_corners = transformed_corners.reshape(-1, 2)        
        bbox = self.bounding_box(transformed_corners)
        bbox_w = bbox[2][0] - bbox[0][0]
        bbox_h = bbox[2][1] - bbox[0][1]

        # Generate random offsets to move the square within the image bounds
        x_offset_range = abs(int(self.width / 2 - bbox_w / 2))
        y_offset_range = abs(int(self.height / 2 - bbox_h / 2))
        offset_x = random.randint(-x_offset_range, x_offset_range)
        offset_y = random.randint(-y_offset_range, y_offset_range)

        # # Apply the random offsets to the 2D points
        square_2d[:, 0] += offset_x
        square_2d[:, 1] += offset_y

        homography_matrix = cv2.getPerspectiveTransform(original_tag, square_2d)
        resized_square = cv2.resize(tag, (tag_size, tag_size), interpolation=cv2.INTER_NEAREST)
        transformed_tag = cv2.warpPerspective(resized_square, homography_matrix, (self.width, self.height))
        transformed_corners = cv2.perspectiveTransform(original_corners, homography_matrix)
        transformed_corners = transformed_corners.reshape(-1, 2)

        # Create a mask from the transformed square
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(transformed_corners)], 255)

        # Use the mask to combine the transformed square with the target image
        img_with_tag = cv2.bitwise_and(self.image, self.image, mask=cv2.bitwise_not(mask))
        img_with_tag = cv2.add(img_with_tag, cv2.bitwise_and(transformed_tag, transformed_tag, mask=mask))
        num_black_pixels = 0
        num_white_pixels = 0
        percent = 0

        # Transform all corners
        for ID, corners in corner_dict.items():
            scaled_corners = [[x * tag_size, y * tag_size] for x, y in corners]
            scaled_corners = np.array(scaled_corners, dtype=np.float32)
            scaled_corners = np.array([scaled_corners])
            corner_dict[ID] = cv2.perspectiveTransform(scaled_corners, homography_matrix)
    

        if self.reflection == True:
            num_black_pixels = np.sum((img_with_tag == 0) & (mask == 255))
            reflected_surface = np.where(img_with_tag <= 50)
            reflection = cv2.imread(self.refpath, cv2.IMREAD_GRAYSCALE)
            
            # Resize reflection image
            reflection = cv2.resize(reflection, (int(tag_size), int(tag_size)), interpolation=cv2.INTER_NEAREST)
            resize_factor = random.uniform(0.5, 4)  
            reflection_resized = cv2.resize(reflection, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)

            h_obj, w_obj = reflection_resized.shape
            h_bg, w_bg = img_with_tag.shape
            min_x = int(np.min(transformed_corners[:, 0]))
            min_y = int(np.min(transformed_corners[:, 1]))
            max_x = int(np.max(transformed_corners[:, 0]))
            max_y = int(np.max(transformed_corners[:, 1]))

            # Randomly place reflection image near black region
            if len(reflected_surface[0]) > 0:
                base_y, base_x = int(transformed_corners[0][1]), int(transformed_corners[0][0])

                offset_range = 10 
                offset_y = random.randint(-offset_range, offset_range)
                offset_x = random.randint(-offset_range, offset_range)
                top_left_y = base_y + offset_y
                top_left_x = base_x + offset_x
                
                # Ensure the object doesn't go out of bounds
                top_left_y = np.clip(top_left_y, 0, h_bg - h_obj)
                top_left_x = np.clip(top_left_x, 0, w_bg - w_obj)
                if min_x <= w_obj and min_y <= h_obj:
                    x_start = max(min_x, 0)
                    x_end = min(max_x, w_obj, self.width)
                    y_start = max(min_y, 0)
                    y_end = min(max_y, h_obj, self.height)
                    for x in range(x_start, x_end):
                        for y in range(y_start, y_end):
                                if reflection_resized[y, x] >= 200: # Reflect white part of the object
                                    if img_with_tag[y, x] == 0:
                                        img_with_tag[y, x] = reflection_resized[y, x]
                                        num_white_pixels += 1

                    img_with_tag = cv2.morphologyEx(img_with_tag, cv2.MORPH_CLOSE, (3,3))
                    percent = num_white_pixels / num_black_pixels * 100


        return img_with_tag, homography_matrix, transformed_corners, corner_dict, percent

    def bounding_box(self, corners):
        # Calculate the minimum and maximum x and y coordinates
        x_min = np.min(corners[:, 0])
        x_max = np.max(corners[:, 0])
        y_min = np.min(corners[:, 1])
        y_max = np.max(corners[:, 1])

        # Calculate the corners of the bounding box
        bounding_box_corners = np.array([
            [x_min, y_min],  
            [x_min, y_max],  
            [x_max, y_max],  
            [x_max, y_min]   
        ])

        return bounding_box_corners
    
    def is_bbox_in_bounds(self, bbox_corners):
        for x, y in bbox_corners:
            if not (0 <= x <= self.width and 0 <= y <= self.height):
                return False  # Out of bounds
        return True  # Fully within bounds

    def extract_inner_corners(self, corners):
        dst_points = np.array([
            [0, 0],   # top-left corner
            [1, 0],   # top-right corner
            [1, 1],   # bottom-right corner
            [0, 1]    # bottom-left corner
        ], dtype=np.float32)

        H, _ = cv2.findHomography(dst_points, corners)

        # Generate the 14x14 grid points in the normalized space
        num_cells = 14
        grid_points_normalized = np.zeros((num_cells + 1, num_cells + 1, 2), dtype=np.float32)

        for i in range(num_cells + 1):
            for j in range(num_cells + 1):
                grid_points_normalized[i, j] = [i / num_cells, j / num_cells]

        grid_points_normalized_flat = grid_points_normalized.reshape(-1, 2)

        # Apply the homography to the grid points
        grid_points_transformed = cv2.perspectiveTransform(np.array([grid_points_normalized_flat]), H)[0]

        inner_corners = np.array([grid_points_transformed[(num_cells+1)*2+2], grid_points_transformed[(num_cells+1)*3-3],
                        grid_points_transformed[(num_cells+1)*12+2], grid_points_transformed[(num_cells+1)*13-3]])

        return inner_corners
    
    def apply_spot_light_effect(self, image):

        center = (random.randint(0, self.width), random.randint(0, self.height))

        if self.radius % 2 == 0:
            self.radius += 1

        mask = np.zeros((self.height, self.width, 1), dtype=np.uint8)
        cv2.circle(mask, center, self.radius, (255, 255, 255), -1)

        # Ensure the kernel size is a tuple of odd integers
        kernel_size = (self.radius, self.radius)
        mask = cv2.GaussianBlur(mask, kernel_size, 0)

        spotlight_image = cv2.addWeighted(image, self.intensity, mask, 1 - self.intensity, 0)
        return spotlight_image

    def add_gaussian_blur(self, image):
        gaussian_blur = cv2.GaussianBlur(image, (self.ksize, self.ksize), self.sigX)
        return gaussian_blur

    def add_gaussian_noise(self, image):
        gauss = np.random.normal(self.mean, self.sigma, (self.height, self.width))
        gauss = gauss.reshape(self.height, self.width)
        noisy = image + gauss
        return noisy

    def apply_motion_blur(self, image):
        kernel_motion_blur = np.zeros((self.motion_blur_size, self.motion_blur_size))
        kernel_motion_blur[int((self.motion_blur_size-1)/2), :] = np.ones(self.motion_blur_size)
        kernel_motion_blur = kernel_motion_blur / self.motion_blur_size

        motion_blur = cv2.filter2D(image, -1, kernel_motion_blur)
        return motion_blur
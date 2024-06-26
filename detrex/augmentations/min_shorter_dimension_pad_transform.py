import numpy as np
from detectron2.data.transforms import Augmentation, Transform, NoOpTransform

class MinShorterDimensionPadTransform(Transform):
    def __init__(self, min_shorter_dim):
        super().__init__()
        self.min_shorter_dim = min_shorter_dim

    def apply_image(self, img):
        height, width = img.shape[:2]

        pad_right = max(0, self.min_shorter_dim - width)
        pad_bottom = max(0, self.min_shorter_dim - height)

        if pad_right > 0 or pad_bottom > 0:
            if img.ndim == 3:  # Color image
                padded_img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
            else:  # Grayscale image
                padded_img = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
        else:
            padded_img = img
        
        return padded_img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        height, width = segmentation.shape[:2]

        pad_right = max(0, self.min_shorter_dim - width)
        pad_bottom = max(0, self.min_shorter_dim - height)

        if pad_right > 0 or pad_bottom > 0:
            padded_segmentation = np.pad(segmentation, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
        else:
            padded_segmentation = segmentation
        
        return padded_segmentation

class MinShorterDimensionPadAugmentation(Augmentation):
    def __init__(self, min_shorter_dim):
        super().__init__()
        self.min_shorter_dim = min_shorter_dim

    def get_transform(self, image):
        height, width = image.shape[:2]
        if height >= self.min_shorter_dim and width >= self.min_shorter_dim:
            return NoOpTransform()
        return MinShorterDimensionPadTransform(self.min_shorter_dim)
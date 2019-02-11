import os
import skimage
import numpy as np
from mrcnn import config as config
from mrcnn import utils as utils

# Defined by images used for training the weights
IMAGE_SIZE = (512,512)

class InferenceConfig(config.Config):
    NAME = "Inference"
    BATCH_SIZE = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = min(IMAGE_SIZE)
    IMAGE_MAX_DIM = max(IMAGE_SIZE)
    DETECTION_MAX_INSTANCES = 500
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.5

    def __init__(self):
        config.Config.__init__(self)

class Dataset(utils.Dataset):
    def __init__(self):
        utils.Dataset.__init__(self)
        self.orig_size = {}
        self.tile_coords = {}

    def load_files(self, files):
        self.add_class("segmentation", 1, "cell")
        for f in files:
            image_id = os.path.basename(f).split('.')[0]
            self.add_image("segmentation",
                           image_id=image_id,
                           path=f)

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        self.orig_size[image_id] = image.shape[:2]
        tiles = self.crop_tiles(image_id, image)
        return tiles

    def crop_tiles(self, image_id, image):
        # All tiles need to be IMAGE_SIZE for prediction
        tile_coords = []
        tiles = []
        if image.shape[0] > IMAGE_SIZE[0] or image.shape[1] > IMAGE_SIZE[1]:
            for y in range(0, image.shape[0], IMAGE_SIZE[0]):
                for x in range(0, image.shape[1], IMAGE_SIZE[1]):
                    xcoord = x
                    ycoord = y
                    if xcoord + IMAGE_SIZE[1] >= image.shape[1]:
                        xcoord = image.shape[1] - IMAGE_SIZE[1]
                    if ycoord + IMAGE_SIZE[0] >= image.shape[0]:
                        ycoord = image.shape[0] - IMAGE_SIZE[0]
                    tile_coords.append((ycoord,xcoord))                    
        else:
            tile_coords.append((0,0))

        for t in tile_coords:
            tile = image[t[0]:t[0]+IMAGE_SIZE[0], t[1]:t[1]+IMAGE_SIZE[1], :]
            tiles.append(tile)
        
        self.tile_coords[image_id] = tile_coords
        return tiles

    def merge_tiles(self, image_id, tile_masks):
        orig_size = self.get_orig_size(image_id)
        mask_img = np.zeros(orig_size, dtype=np.uint8)
        tile_coords = self.tile_coords[image_id]
        for tile,coords in zip(tile_masks, tile_coords):
            mask_img[coords[0]:coords[0]+IMAGE_SIZE[0], coords[1]:coords[1]+IMAGE_SIZE[1]] = tile
        
        return mask_img

    def get_orig_size(self, image_id):
        return self.orig_size.get(image_id, (0,0))

    def resize_to_orig_size(self, image_id, mask):
        mask = skimage.transform.resize(mask, self.get_orig_size(image_id))
        mask[mask > 0] = 1
        return mask

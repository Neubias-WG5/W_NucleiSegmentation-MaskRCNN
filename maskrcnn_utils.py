import os
from operator import itemgetter
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
    RPN_NMS_THRESHOLD = 0.7
    POST_NMS_ROIS_INFERENCE = 1000
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

    def load_image(self, image_id, tile_overlap = 0):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        self.orig_size[image_id] = image.shape[:2]
        tiles = self.crop_tiles(image_id, image, tile_overlap)
        return tiles

    def crop_tiles(self, image_id, image, tile_overlap = 0):
        # All tiles need to be IMAGE_SIZE for prediction
        if tile_overlap > max(IMAGE_SIZE) / 2:
            tile_overlap = max(IMAGE_SIZE) / 2

        tile_coords = []
        tiles = []
        if image.shape[0] > IMAGE_SIZE[0] or image.shape[1] > IMAGE_SIZE[1]:
            for y in range(0, image.shape[0], IMAGE_SIZE[0]-tile_overlap):
                for x in range(0, image.shape[1], IMAGE_SIZE[1]-tile_overlap):
                    xcoord = x
                    ycoord = y
                    if (xcoord + IMAGE_SIZE[1] >= image.shape[1]) and (xcoord != 0):
                        xcoord = image.shape[1] - IMAGE_SIZE[1]
                    if (ycoord + IMAGE_SIZE[0] >= image.shape[0]) and (ycoord != 0):
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
        label_img = np.zeros(orig_size, dtype=np.uint16)
        tile_coords = self.tile_coords[image_id]
        objects = []
        for tile,coords in zip(tile_masks, tile_coords):
            for i in range(tile['rois'].shape[0]):
                m_coords = tile['rois'][i]
                mask = tile['masks'][:,:,i]
                mrows = np.any(mask, axis=1)
                mcols = np.any(mask, axis=0)
                try:
                    rmin,rmax = np.where(mrows)[0][[0, -1]]
                    cmin,cmax = np.where(mcols)[0][[0, -1]]
                except:
                    continue
                
                mask = mask[rmin:rmax+1, cmin:cmax+1].astype(np.uint16)
                m_coords[0] = coords[0] + rmin
                m_coords[2] = coords[0] + rmax
                m_coords[1] = coords[1] + cmin
                m_coords[3] = coords[1] + cmax
                objects.append({'roi': m_coords,
                                'score': tile['scores'][i],
                                'mask': mask})

        objects = sorted(objects, key=itemgetter('score'), reverse=True)
        objnum = 0
        for obj in objects:
            roi = obj['roi']
            mcrop = label_img[roi[0]:roi[2]+1, roi[1]:roi[3]+1]
            if ~((mcrop.astype(np.bool) & obj['mask'].astype(np.bool)).any()):
                objnum += 1
                (obj['mask'])[obj['mask'] > 0] = objnum
                mcrop = mcrop + obj['mask']
                label_img[roi[0]:roi[2]+1, roi[1]:roi[3]+1] = mcrop
        
        return label_img

    def get_orig_size(self, image_id):
        return self.orig_size.get(image_id, (0,0))

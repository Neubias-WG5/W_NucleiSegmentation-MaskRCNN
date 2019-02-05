import sys
import os
from maskrcnn_utils import InferenceConfig
from maskrcnn_utils import Dataset
from mrcnn import model as modellib
from cytomine.models import Job
from neubiaswg5 import CLASS_OBJSEG
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics


def main(argv):
    base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
    problem_cls = CLASS_OBJSEG

    with NeubiasJob.from_cli(argv) as nj:
        nj.job.update(status=Job.RUNNING, progress=0, statusComment="Initialisation...")
        # 1. Prepare data for workflow
        in_imgs, gt_imgs, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, nj, is_2d=True, **nj.flags)

        working_path = os.path.join(base_path, "data", str(nj.job.id))
        gt_suffix = "_lbl"

        # 2. Run Mask-RCNN prediction
        nj.job.update(progress=25, statusComment="Launching workflow...")

        files = [os.path.join(in_path,"{}.tif".format(image.id)) for image in in_imgs]
        model_dir = "/Mask_RCNN/logs"
        dataset = Dataset()
        dataset.load_files(files)
        dataset.prepare()
        inference_config = InferenceConfig()
        model = modellib.MaskRCNN(mode = "inference",
                                  config = inference_config,
                                  model_dir = model_dir)
        model.load_weights(os.path.join(model_dir,'weights.h5'), by_name=True)
        for i,image_id in enumerate(dataset.image_ids):
            tiles = dataset.load_image(image_id)
            orig_size = dataset.get_orig_size(image_id)
            mask_img = np.zeros(orig_size, dtype=np.uint8)

            tile_masks = []
            for image in tiles:
                tile_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = model.detect([image], verbose=0)[0]
                for m in range(mask['masks'].shape[2]):
                    objmask = mask['masks'][:,:,m]
                    objmask = objmask.astype(np.uint8)
                    objmask = skimage.morphology.binary_erosion(objmask, skimage.morphology.disk(1))
                    tile_mask[objmask > 0] = 255
            tile_masks.append(tile_mask)
    
            mask_img = dataset.merge_tiles(image_id, tile_masks)
            skimage.io.imsave(files[i], mask_img)

        # 3. Upload data to Cytomine
        upload_data(problem_cls, nj, in_imgs, out_path, **nj.flags, monitor_params={
            "start": 60, "end": 90, "period": 0.1,
            "prefix": "Extracting and uploading polygons from masks"})
        
        # 4. Compute and upload metrics
        nj.job.update(progress=90, statusComment="Computing and uploading metrics...")
        upload_metrics(problem_cls, nj, in_imgs, gt_path, out_path, tmp_path, **nj.flags)

        # 5. Pipeline finished
        nj.job.update(progress=100, status=Job.TERMINATED, status_comment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])

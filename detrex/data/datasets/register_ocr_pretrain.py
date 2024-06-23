from detectron2.data.datasets import register_coco_instances
import os
import logging

PRETRAIN_NAME_TO_SPLITS = {
    'DDI_100': ['train', 'test']
}


def register_pretrain(datasets_root: str):
    logger = logging.getLogger("detectron2")
    for ds_name, ds_split in PRETRAIN_NAME_TO_SPLITS.items():
        for split in ds_split:
            json_annots_path = os.path.join(
                datasets_root, ds_name, f'{ds_name}_{split}.json')
            image_paths = os.path.join(
                datasets_root, ds_name, f'{split}_images')
            if not os.path.exists(json_annots_path):
                logger.warn(
                    f'Could not find {ds_name}_{split} json path (expected {json_annots_path})')
                continue
            if not os.path.exists(image_paths):
                logger.warn(
                    f'Could not find {ds_name}_{split} image path (expected {image_paths})')
                continue
            
            logger.info(f"Registering {ds_name}_{split}")

            register_coco_instances(f'{ds_name}_{split}',
                                    metadata={},
                                    json_file=json_annots_path,
                                    image_root=image_paths)

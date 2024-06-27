from .text import register_text_instances
import os


PRETRAIN_NAME_TO_SPLITS = {
    'DDI_100': ['train', 'test'],
    'Arshab_7k': ['train'],
    'Hiertext': ['train', 'validation'],
    'ic15_clean': ['train', 'test'],
    'ic17mlt_clean': ['train', 'val'],
    'ic19mlt_clean': ['train'],
    'textocr': ['train']

}


def register_pretrain(datasets_root: str):
    for ds_name, ds_split in PRETRAIN_NAME_TO_SPLITS.items():
        for split in ds_split:
            full_ds_name = f'{ds_name}_{split}'

            register_text_instances(
                full_ds_name,
                metadata={},
                json_file=os.path.join(
                    datasets_root, ds_name, f'{full_ds_name}.json'),
                image_root=os.path.join(
                    datasets_root, ds_name, f'{split}_images'),
            )


_root = os.getenv("DETECTRON2_DATASETS",
                  "./datasets/")
register_pretrain(_root)

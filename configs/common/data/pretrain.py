from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detrex.data import DetrDatasetMapper, get_dataset_dicts_and_sampler, build_weighted_detection_train_loader
from detrex.data.datasets.register_ocr_pretrain import register_pretrain
from detrex.evaluation.multi_evaluator import create_evaluator

dataloader = OmegaConf.create()


dataloader.train = L(build_weighted_detection_train_loader)(
    dataset=L(get_dataset_dicts_and_sampler)(
        names=["DDI_100_train",
               "Arshab_7k_train",
               'Hiertext_train',
               'ic15_clean_train',
               'ic17mlt_clean',
               'ic19_mlt_clean',
               'textocr_train'],
        weights=[0.2]  # uniform
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.RandomRotation)(
                angle=(0, 0, 0, 90, 180, 270),
                sample_style='choice'
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.RandomRotation)(
                angle=(0, 0, 0, 90, 180, 270),
                sample_style='choice'
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names=["DDI_100_test", 'Hiertext_validation',
               'ic15_clean_test', 'ic17mlt_val'],
        filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(create_evaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=2000,
    output_dir='./eval',
    vis=True,
    coco=False
)
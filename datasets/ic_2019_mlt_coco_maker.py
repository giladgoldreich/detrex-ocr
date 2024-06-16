import sys
sys.path.append('.')
sys.path.append('..')
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Tuple, List, Dict, Optional, Any
import warnings
import numpy as np
from PIL import Image, ImageOps
import pyrallis
import pickle
from sklearn.model_selection import train_test_split
import cv2
import json
from enum import Enum, auto
from datasets.ic_2017_mlt_coco_maker import IC17MLTCocoMaker, IC17MLTCocoMakingConfig, OtherLangugePolicy, IgnorePolicies, ImageWithAnnots


@dataclass
class IC19MLTCocoMakingConfig(IC17MLTCocoMakingConfig):
    dataset_name: ClassVar[str] = 'ic19mlt_clean'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/ICDAR2019_mlt')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://rrc.cvc.uab.es/?ch=15&com=introduction",
        "version": "1.0",
        "year": 2019,
        "contributor": "ICDAR",
        "date_created": "2019/01/01"
    })
    test_run: bool = False
    languages: List[str] = field(default_factory=lambda: ['Arabic', 'Latin'])
    other_language_policy: OtherLangugePolicy = OtherLangugePolicy.MARK_IGNORE
    subset_name_to_gt_dir: Dict[str, str] = field(default_factory=lambda: 
        {
            'train': 'train_gt_t13',
        })
    skip_image_without_lang_annots: bool = True
    gt_file_prefix: Optional[str] = None
    ignore_policy: IgnorePolicies = IgnorePolicies.SKIP_IMAGE
    subset_name_to_max_images: Dict[str, int] = field(default_factory=lambda: {
        'train': 2000
    })
    

@dataclass
class IC19MLTCocoMaker(IC17MLTCocoMaker):
    config: IC19MLTCocoMakingConfig
    
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        img_file_path = self.image_paths[idx]
        subset = self.subsets[idx]
        image_num = int(img_file_path.stem.split('_')[-1])
        if subset in self.config.subset_name_to_max_images and image_num > self.config.subset_name_to_max_images[subset]:
            return None
        return super().__getitem__(idx)
        
        

def run(cfg: IC19MLTCocoMakingConfig):
    IC19MLTCocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: IC19MLTCocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
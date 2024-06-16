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
from datasets.dataset_maker import DatasetMakingConfig, CocoDatasetMaker, ImageWithAnnots, IgnorePolicies



@dataclass
class IC15CocoMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'ic15_full'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/ICDAR2015')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://rrc.cvc.uab.es/?ch=4&com=introduction",
        "version": "1.0",
        "year": 2015,
        "contributor": "ICDAR",
        "date_created": "2015/01/01"
    })
    test_run: bool = False
    ignore_policy: IgnorePolicies = IgnorePolicies.KEEP    

@dataclass
class IC15CocoMaker(CocoDatasetMaker):
    config: IC15CocoMakingConfig
    all_annot_dicts: List[Dict] = field(init=False)
    image_paths: List[Path] = field(init=False)
    gt_paths: List[Path] = field(init=False)
    subsets: List[str] = field(init=False)
    
    def __post_init__(self):
        print('Loading annots')
        train_images = list((self.config.dataset_root / 'ch4_training_images').glob('*.jpg'))
        train_gts = [p.parent.parent / 'ch4_training_localization_transcription_gt' / f'gt_{p.stem}.txt' for p in train_images]
        
        test_images = list((self.config.dataset_root / 'ch4_test_images').glob('*.jpg'))
        test_gts = [p.parent.parent / 'ch4_test_localization_transcription_gt' / f'gt_{p.stem}.txt' for p in test_images]
        
        self.image_paths = train_images + test_images
        self.gt_paths = train_gts + test_gts
        self.subsets = ['train'] * len(train_images) + ['test'] * len(test_images)
                            
    def get_number_of_images(self) -> int:
        return len(self.image_paths)
    
        
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        
        im_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        save_name = f'{im_path.parent.name}_{im_path.name}'
        all_xy_coords = []
        words = []
        ignores = []
        with open(gt_path, 'r') as f:
            raw_annots = f.readlines()
        for ann in raw_annots:
            ann = ann.replace('\ufeff', '').replace('\n', '')
            split_by_comma = ann.split(',')
            if len(split_by_comma) < 9:
                print(ann)
                continue
            cur_xyxy = list(map(int, split_by_comma[:8]))
            all_xy_coords.append(np.array(cur_xyxy).reshape(4, 2))
            cur_text = split_by_comma[8]
            if cur_text == '' or cur_text == "###":
                cur_ignore = True
            else:
                cur_ignore = False
            words.append(cur_text)
            ignores.append(cur_ignore)

        im = Image.open(im_path)
        
        return ImageWithAnnots(
            im=im,
            xy_coords=all_xy_coords,
            ignore_mask=ignores,
            subset=self.subsets[idx],
            save_name=save_name,
            original_img_path=im_path,
            texts=words
        )
        

def run(cfg: IC15CocoMakingConfig):
    IC15CocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: IC15CocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
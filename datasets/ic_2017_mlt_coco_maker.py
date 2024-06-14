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

class OtherLangugePolicy(Enum):
    KEEP = auto()
    MARK_IGNORE = auto()
    SKIP_IMAGE = auto()
    DELETE = auto()


@dataclass
class IC17MLTCocoMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'ic7mlt_full'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/ICDAR2017_mlt')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://rrc.cvc.uab.es/?ch=4&com=introduction",
        "version": "1.0",
        "year": 2017,
        "contributor": "ICDAR",
        "date_created": "2017/01/01"
    })
    test_run: bool = False
    languages: List[str] = field(default_factory=lambda: ['Arabic', 'Latin'])
    other_language_policy: OtherLangugePolicy = OtherLangugePolicy.MARK_IGNORE
    subset_name_to_gt_dir: Dict[str, str] = field(default_factory=lambda: 
        {
            'train': 'ch8_training_localization_transcription_gt_v2',
            'val': 'ch8_validation_localization_transcription_gt_v2'
        })
    gt_file_prefix: Optional[str] = 'gt'
    skip_image_without_lang_annots: bool = True
    ignore_policy: IgnorePolicies = IgnorePolicies.KEEP
    

@dataclass
class IC17MLTCocoMaker(CocoDatasetMaker):
    config: IC17MLTCocoMakingConfig
    image_paths: List[Path] = field(init=False)
    gt_paths: List[Path] = field(init=False)
    subsets: List[str] = field(init=False)
    
    def __post_init__(self):
        subsets = []
        image_paths = []
        gt_paths = []
        for subset, subset_gt_dir in self.config.subset_name_to_gt_dir.items():
            subset_images = sorted(list((self.config.dataset_root / subset).rglob('*.jpg')))
            subset_gts = []
            prefix = f'{self.config.gt_file_prefix}_' if self.config.gt_file_prefix else ''
            subset_gts = [p.parent.parent / subset_gt_dir / f'{prefix}{p.stem}.txt' for p in subset_images]
            image_paths.extend(subset_images)
            gt_paths.extend(subset_gts)
            subsets.extend([subset] * len(subset_images))
        self.subsets = subsets
        self.image_paths = image_paths
        self.gt_paths = gt_paths
            
                            
    def get_number_of_images(self) -> int:
        return len(self.image_paths)
    
        
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        
        im_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]
        save_name = f'{im_path.parent.name}_{im_path.name}'
        xywh_bboxes = []
        words = []
        ignores = []
        with open(gt_path, 'r') as f:
            raw_annots = f.readlines()
        
        num_annots_in_lang = 0
        num_annots_in_other_lang = 0
        for ann in raw_annots:
            ann = ann.replace('\ufeff', '').replace('\n', '')
            split_by_comma = ann.split(',')
            if len(split_by_comma) < 10:
                print(ann)
                continue
            cur_xyxy = list(map(int, split_by_comma[:8]))
            cur_xmin = min(cur_xyxy[::2])
            cur_ymin = min(cur_xyxy[1::2])
            cur_xmax = max(cur_xyxy[::2])
            cur_ymax = max(cur_xyxy[1::2])
            
            cur_lang = split_by_comma[8]
            cur_text = split_by_comma[9]
            if cur_text == '' or cur_text == "###":
                cur_ignore = True
            else:
                cur_ignore = False
            
            if cur_lang not in self.config.languages:
                num_annots_in_other_lang += 1
                if self.config.other_language_policy == OtherLangugePolicy.KEEP:
                    pass
                elif self.config.other_language_policy == OtherLangugePolicy.MARK_IGNORE:
                    cur_ignore = True
                elif self.config.other_language_policy == OtherLangugePolicy.DELETE:
                    continue
                elif self.config.other_language_policy == OtherLangugePolicy.SKIP_IMAGE:
                    return None
                else:
                    raise RuntimeError(self.config.other_language_policy)
            elif cur_lang in self.config.languages:
                num_annots_in_lang += 1
            
            xywh_bboxes.append([cur_xmin, cur_ymin, cur_xmax-cur_xmin, cur_ymax-cur_ymin])
            words.append(cur_text)
            ignores.append(cur_ignore)
            
        if num_annots_in_lang == 0 and num_annots_in_other_lang > 0 and self.config.skip_image_without_lang_annots:
            return None

        im = Image.open(im_path)
        
        return ImageWithAnnots(
            im=im,
            xywh_boxes=np.array(xywh_bboxes).astype(float),
            ignore_mask=ignores,
            subset=self.subsets[idx],
            save_name=save_name,
            original_img_path=im_path,
            texts=words
        )
        

def run(cfg: IC17MLTCocoMakingConfig):
    IC17MLTCocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: IC17MLTCocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
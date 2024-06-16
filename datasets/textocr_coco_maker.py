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

class HardAnnotPolicy(Enum):
    KEEP = auto()
    MARK_IGNORE = auto()
    DELETE = auto()


@dataclass
class TextOCRCocoMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'textocr'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/TextOCR')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://textvqa.org/textocr/dataset/",
        "version": "0.1",
        "year": 2021,
        "contributor": "TextOCR",
        "date_created": "2021/01/01"
    })
    test_run: bool = False
    ignore_policy: IgnorePolicies = IgnorePolicies.SKIP_IMAGE
    max_angle: int = 15
    high_angle_policy: HardAnnotPolicy = HardAnnotPolicy.MARK_IGNORE
    

@dataclass
class TextOCRCocoMaker(CocoDatasetMaker):
    config: TextOCRCocoMakingConfig
    annot_dict: Dict[str, Dict[str, List[Any]]] = field(init=False)
    img_ids: List[str] = field(init=False)
    
    def __post_init__(self):
        json_annot_file = self.config.dataset_root / 'TextOCR_0.1_train.json'
        print('reading', json_annot_file)
        with open(json_annot_file, 'r') as f:
            self.annot_dict = json.load(f)
        self.img_ids = list(self.annot_dict['imgs'].keys())
                            
    def get_number_of_images(self) -> int:
        return len(self.img_ids)
    
        
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        
        im_id = self.img_ids[idx]
        img_dict = self.annot_dict['imgs'][im_id]

        subset = img_dict['set']
        origin_img_path = self.config.dataset_root / f'{subset}_images' / f'{im_id}.jpg'
        im = Image.open(origin_img_path)
        save_name = origin_img_path.name
        
        ann_ids_list = self.annot_dict['imgToAnns'][im_id]
        ann_dicts = [self.annot_dict['anns'][ann_id] for ann_id in ann_ids_list]
        xywh_boxes = np.array([d['bbox'] for d in ann_dicts]).astype(float)
        words = np.array([d['utf8_string'] for d in ann_dicts])
        
        ignore_mask = np.zeros(len(xywh_boxes), dtype=bool)
        keep_mask = np.ones(len(xywh_boxes), dtype=bool)
        
        xy_coords = [np.array(d['points']).reshape(-1, 2) for d in ann_dicts]
        cxcywha_boxes = self.xy_coords_to_cxcywha_boxes(xy_coords)
        alpha = cxcywha_boxes[:, 4]
        too_high_angle_mask = (np.abs(alpha) >= self.config.max_angle) & (np.abs(alpha) <= 90 - self.config.max_angle)
        
        if self.config.high_angle_policy == HardAnnotPolicy.KEEP:
            pass
        elif self.config.high_angle_policy == HardAnnotPolicy.MARK_IGNORE:
            ignore_mask[too_high_angle_mask] = True
        elif self.config.high_angle_policy == HardAnnotPolicy.DELETE:
            keep_mask[too_high_angle_mask] = False
        else:
            raise RuntimeError(self.config.high_angle_policy)    
        
        return ImageWithAnnots(
            im=im,
            xy_coords=xy_coords,
            ignore_mask=ignore_mask[keep_mask],
            subset=subset,
            save_name=save_name,
            original_img_path=origin_img_path,
            texts=words[keep_mask],
            # img_id=im_id,
            # annot_ids=ann_ids_list
        )
        

def run(cfg: TextOCRCocoMakingConfig):
    TextOCRCocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: TextOCRCocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
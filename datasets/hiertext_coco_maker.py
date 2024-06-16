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
from datasets.dataset_maker import DatasetMakingConfig, CocoDatasetMaker, ImageWithAnnots


class HardAnnotPolicy(Enum):
    KEEP = auto()
    MARK_IGNORE = auto()
    DELETE = auto()
    

@dataclass
class HierTextCocoMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'Hiertext'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/hiertext')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://github.com/google-research-datasets/hiertext",
        "version": "1.0",
        "year": 2022,
        "contributor": "Google",
        "date_created": "2022/06/03"
    })
    test_run: bool = False
    max_angle: int = 20
    angle_policy: HardAnnotPolicy = HardAnnotPolicy.MARK_IGNORE
    handwriting_policy: HardAnnotPolicy = HardAnnotPolicy.KEEP
    legible_policy: HardAnnotPolicy = HardAnnotPolicy.MARK_IGNORE
    

@dataclass
class HierTextCocoMaker(CocoDatasetMaker):
    config: HierTextCocoMakingConfig
    all_annot_dicts: List[Dict] = field(init=False)
    image_sources: List[str] = field(init=False)
    
    def __post_init__(self):
        print('loading train annots...')
        with open(self.config.dataset_root / 'train.jsonl', 'r') as f:
            train_annots = json.load(f)['annotations']        
        
        print('loading validation annots...')
        with open(self.config.dataset_root / 'validation.jsonl', 'r') as f:
            validation_annots = json.load(f)['annotations']
            
        self.image_sources = ['train'] * len(train_annots) + ['validation'] * len(validation_annots)
        self.all_annot_dicts = train_annots + validation_annots
            
    def get_number_of_images(self) -> int:
        return len(self.all_annot_dicts)
    
        
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        
        cur_annot = self.all_annot_dicts[idx]
        image_id = cur_annot['image_id']
        subset = self.image_sources[idx]
        im_path = self.config.dataset_root / subset / f'{image_id}.jpg'
        im = Image.open(im_path)
        file_name = im_path.name
        
        word_annots = []
        for p_annot in cur_annot['paragraphs']:
            for line_annot in p_annot['lines']:
                word_annots.extend(line_annot['words'])
        
        texts = np.array([d['text'] for d in word_annots])
        all_xyxy_coords = [np.array(d['vertices']).astype(float) for d in word_annots]
        is_legibles = np.array([d['legible'] for d in word_annots]).astype(bool)
        is_handwritten = np.array([d['vertical'] for d in word_annots]).astype(bool)
        cxcywha_boxes = [cv2.minAreaRect(xyxy.astype(int)) for xyxy in all_xyxy_coords]
        cx = np.array([t[0][0] for t in cxcywha_boxes]).astype(float)
        cy = np.array([t[0][1] for t in cxcywha_boxes]).astype(float)
        w = np.array([t[1][0] for t in cxcywha_boxes]).astype(float)
        h = np.array([t[1][1] for t in cxcywha_boxes]).astype(float)
        alpha = np.array([t[2] for t in cxcywha_boxes]).astype(float)
        too_high_angle_mask = (np.abs(alpha) >= self.config.max_angle) & (np.abs(alpha) <= 90 - self.config.max_angle)
        
        
        cos_a = np.abs(np.cos(alpha * np.pi / 180.))
        sin_a = np.abs(np.sin(alpha * np.pi / 180.))
        
        new_w = cos_a * w + sin_a * h
        new_h = cos_a * h + sin_a * w
        xmin = cx - new_w / 2
        ymin = cy - new_h / 2
        
        xywh_bboxes = np.stack([xmin, ymin, new_w, new_h], axis=-1)
        
        ignore_mask = np.zeros(len(texts), dtype=bool)
        keep_mask = np.ones(len(texts), dtype=bool)
        for msk, policy in zip([too_high_angle_mask, ~is_legibles, is_handwritten],
                               [self.config.angle_policy, self.config.legible_policy, self.config.handwriting_policy]):
            if policy == HardAnnotPolicy.KEEP:
                continue
            elif policy == HardAnnotPolicy.MARK_IGNORE:
                ignore_mask[msk] = True
            elif policy == HardAnnotPolicy.DELETE:
                keep_mask[msk] = False
            else: 
                raise ValueError(policy)
        

        xywh_bboxes = xywh_bboxes[keep_mask]
        ignore_mask = ignore_mask[keep_mask]
        texts = texts[keep_mask]
        return ImageWithAnnots(im=im,
                               img_id=image_id, 
                               xywh_boxes=xywh_bboxes,
                               ignore_mask=ignore_mask,
                               subset=subset,
                               save_name=im_path.name,
                               texts=texts,
                               original_img_path=im_path)        
            

def run(cfg: HierTextCocoMakingConfig):
    HierTextCocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: HierTextCocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
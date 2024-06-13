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

from datasets.dataset_maker import DatasetMakingConfig, CocoDatasetMaker, XYWH_BOXES

@dataclass
class DDI100CocoMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'DDI_100'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/dataset_v1.3')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://github.com/machine-intelligence-laboratory/DDI-100/",
        "version": "1.3",
        "year": 2019,
        "contributor": "Ilia Zharikov",
        "date_created": "2021/12/25"
    })
    blend: bool = True
    blend_ratio: float = 0.6
    train_ratio: float = 0.85
    blend_if_background_not_exists: bool = False
    random_state: int = 42
    only_imgs_with_background: bool = True
    

@dataclass
class DDI100CocoMaker(CocoDatasetMaker):
    config: DDI100CocoMakingConfig
    all_origin_pkl_files: List[Path] = field(init=False)
    train_mask: np.ndarray = field(init=False)
    
    def __post_init__(self):
        all_origin_pkl_files = sorted(list(self.config.dataset_root.rglob('*.pkl')))
        if self.config.only_imgs_with_background:
            all_background_img_files = [self.extract_background_img_file_from_pkl_file(p) for p in all_origin_pkl_files]
            all_origin_pkl_files = [pf for pf, bif in zip(all_origin_pkl_files, all_background_img_files) if bif.exists()]
        self.all_origin_pkl_files = all_origin_pkl_files
        train_indices, test_indices = train_test_split(list(range(len(self.all_origin_pkl_files))),
                                                       train_size=self.config.train_ratio,
                                                       random_state=self.config.random_state)
        self.train_mask = np.zeros(len(self.all_origin_pkl_files), dtype=bool)
        self.train_mask[np.array(train_indices)] = True
        self.train_mask[np.array(test_indices)] = False
        
    @staticmethod
    def extract_origin_img_file_from_pkl_file(pkl_file_path: Path) -> Path:
        origin_im_file = (pkl_file_path.parent.parent / 'orig_texts' / pkl_file_path.stem).with_suffix('.png')
        return origin_im_file
    
    @staticmethod
    def extract_background_img_file_from_pkl_file(pkl_file_path: Path) -> Path:
        background_im_file = (pkl_file_path.parent.parent / 'orig_backgrounds' / pkl_file_path.stem).with_suffix('.png')
        return background_im_file
            
    def get_number_of_images(self) -> int:
        return len(self.all_origin_pkl_files)
    
    
    @staticmethod
    def read_fake_rgb_image(im_path: Path) -> Image.Image:
        fake_rgb_im = Image.open(im_path).convert('RGB')
        real_rgb_im = Image.fromarray(np.array(fake_rgb_im)[:, :, ::-1])
        return real_rgb_im
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, XYWH_BOXES, str, str, List[bool], Optional[List[str]]]:
        cur_pkl_file = self.all_origin_pkl_files[idx]        
        
        origin_im_file = self.extract_origin_img_file_from_pkl_file(cur_pkl_file)
        background_im_path = self.extract_background_img_file_from_pkl_file(cur_pkl_file)
        
        if not origin_im_file.exists():
            warnings.warn(f"Cannot find {origin_im_file}")
            return None, [], "none", "none", [], None
        
        with open(cur_pkl_file, 'rb') as f:
            reverse_annots = pickle.load(f)
        
        texts = list(map(lambda d: d['text'], reverse_annots))
        xyxy_coords = list(map(lambda d: d['box'][:, ::-1], reverse_annots))
        xywh_bboxes = list(map(lambda xyxy: [xyxy[:, 0].min(), 
                                             xyxy[:, 1].min(),
                                             xyxy[:, 0].max() - xyxy[:, 0].min(), 
                                             xyxy[:, 1].max() - xyxy[:, 1].min()],
                               xyxy_coords))
        xywh_bboxes = np.array(xywh_bboxes).astype(float)
        ignores = [False] * len(xywh_bboxes)
        file_name = f'part_{origin_im_file.parent.parent.name}_orig_{origin_im_file.stem}'
        origin_im = self.read_fake_rgb_image(origin_im_file)
        if self.config.blend and self.config.blend_ratio > 0 and background_im_path.exists():
            background_im = self.read_fake_rgb_image(background_im_path)
            if origin_im_file.parent.parent.name == '02':
                background_im_arr = np.array(background_im)
                background_im_arr[-300:, :, :] = 255
                background_im = Image.fromarray(background_im_arr)
            
            final_im = Image.blend(background_im, origin_im, self.config.blend_ratio).convert('RGB')            
            file_name = f'{file_name}_blend'
                            
        else:
            final_im = origin_im
            
        file_name = f'{file_name}.png'
        return final_im, xywh_bboxes, file_name, 'train' if self.train_mask[idx] else 'test', ignores, texts
    
    

def run(cfg: DDI100CocoMakingConfig):
    DDI100CocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: DDI100CocoMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
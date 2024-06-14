import sys
sys.path.append('.')
sys.path.append('..')
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Tuple, List, Dict, Optional, Any
import numpy as np
from PIL import Image
import pyrallis
import pandas as pd
import ast


from datasets.dataset_maker import DatasetMakingConfig, CocoDatasetMaker, ImageWithAnnots

@dataclass
class ArshabMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'Arshab_7k'
    dataset_root: Path =Path('/Users/giladgoldreich/Downloads/Arshasb_7k/')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://github.com/persiandataset/Arshasb",
        "version": "1.0",
        "year": 2021,
        "contributor": "Arshab",
        "date_created": "2021/11/24"
    })
    test_run: bool = False

@dataclass
class ArshabCocoMaker(CocoDatasetMaker):
    config: ArshabMakingConfig
    all_labels_xlsx_files: List[Path] = field(init=False)
    
    def __post_init__(self):
        self.all_labels_xlsx_files = sorted(list(self.config.dataset_root.rglob('*label_*.xlsx')))
            
    def get_number_of_images(self) -> int:
        return len(self.all_labels_xlsx_files)
    
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        cur_xl_file = self.all_labels_xlsx_files[idx]
        page_num = cur_xl_file.stem.split('_')[-1]
        im_path = self.config.dataset_root / f'{page_num}/page_{page_num}.png'
        im = Image.open(im_path).convert('RGB')
        annot_df = pd.read_excel(cur_xl_file)
        words = annot_df['word'].tolist()
        x1_arr = annot_df['point1'].apply(ast.literal_eval).apply(lambda xy: xy[0]).values.astype(float)
        y1_arr = annot_df['point1'].apply(ast.literal_eval).apply(lambda xy: xy[1]).values.astype(float)
        x2_arr = annot_df['point4'].apply(ast.literal_eval).apply(lambda xy: xy[0]).values.astype(float)
        y2_arr = annot_df['point4'].apply(ast.literal_eval).apply(lambda xy: xy[1]).values.astype(float)
        w_arr = x2_arr - x1_arr
        h_arr = y2_arr - y1_arr
        xywh_boxes = np.stack([x1_arr, y1_arr, w_arr, h_arr], axis=-1)
        save_name = im_path.name        
        return ImageWithAnnots(im=im,
                               xywh_boxes=xywh_boxes,
                               ignore_mask=[False] * len(xywh_boxes),
                               subset='train',
                               original_img_path=im_path,
                               save_name=im_path.name,
                               texts=words)
    
    

def run(cfg: ArshabMakingConfig):
    ArshabCocoMaker(cfg).create()
    
@pyrallis.wrap()
def main(cfg: ArshabMakingConfig):
    run(cfg)
    
if __name__ == '__main__':
    main()
        
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
from sklearn.model_selection import train_test_split

from datasets.dataset_maker import DatasetMakingConfig, CocoDatasetMaker, ImageWithAnnots

@dataclass
class ArshabMakingConfig(DatasetMakingConfig):
    dataset_name: ClassVar[str] = 'Arshab_7k'
    dataset_root: Path =Path('/nfs/private/gilad/ocr_detection/data/raw/Arshasb_7k/')
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://github.com/persiandataset/Arshasb",
        "version": "1.0",
        "year": 2021,
        "contributor": "Arshab",
        "date_created": "2021/11/24"
    })
    test_run: bool = False
    train_ratio: float = 0.85
    random_state: int = 42

@dataclass
class ArshabCocoMaker(CocoDatasetMaker):
    config: ArshabMakingConfig
    train_mask: np.ndarray = field(init=False)
    all_labels_xlsx_files: List[Path] = field(init=False)
    
    def __post_init__(self):
        self.all_labels_xlsx_files = sorted(list(self.config.dataset_root.rglob('*label_*.xlsx')))
        train_indices, test_indices = train_test_split(list(range(len(self.all_labels_xlsx_files))),
                                                       train_size=self.config.train_ratio,
                                                       random_state=self.config.random_state)
        train_mask = np.zeros(len(self.all_labels_xlsx_files), dtype=bool)
        train_mask[np.array(train_indices)] = True
        train_mask[np.array(test_indices)] = False
        self.train_mask = train_mask
            
    def get_number_of_images(self) -> int:
        return len(self.all_labels_xlsx_files)
    
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        cur_xl_file = self.all_labels_xlsx_files[idx]
        page_num = cur_xl_file.stem.split('_')[-1]
        im_path = self.config.dataset_root / f'{page_num}/page_{page_num}.png'
        im = Image.open(im_path).convert('RGB')
        annot_df = pd.read_excel(cur_xl_file)
        words = annot_df['word'].tolist()
        all_points = np.hstack([
            np.expand_dims(np.array(annot_df[f'point{i}'].apply(ast.literal_eval).apply(lambda l: list(map(float, l))).tolist()),
                           1)
            for i in range(1, 5)
        ]
        )
        all_points = all_points[:, [0, 1, 3, 2]]
        return ImageWithAnnots(im=im,
                               xy_coords=all_points,
                               ignore_mask=[False] * len(all_points),
                               subset='train' if self.train_mask[idx] else 'test',
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
        
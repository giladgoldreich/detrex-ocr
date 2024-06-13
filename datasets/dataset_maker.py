import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable, Tuple, List, Dict, Optional, Any
import abc
from tqdm import tqdm
import warnings
import numpy as np
import json
from PIL import Image
from collections import defaultdict
import pyrallis

XYWH_BOXES = Iterable[Tuple[float, float, float, float]]


class IgnorePolicies(Enum):
    KEEP = auto()
    REMOVE_ANNOT = auto()
    SKIP_IMAGE = auto()
    
class SpacePolicies(Enum):
    KEEP = auto()
    MARK_IGNORE = auto()
    SKIP_IMAGE = auto()
        

@dataclass
class DatasetMakingConfig:
    dataset_name: ClassVar[str]
    
    dataset_root: Path
    destination: Path = Path('./datasets')
    exists_ok: bool = True
    skip_image_without_gt_annots: bool = True
    add_dataset_name_to_destination: bool = True
    ignore_policy: IgnorePolicies = IgnorePolicies.SKIP_IMAGE
    space_policy: SpacePolicies = SpacePolicies.MARK_IGNORE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.add_dataset_name_to_destination:
            self.destination = self.destination / self.dataset_name
            self.add_dataset_name_to_destination = False
    
    
@dataclass
class CocoDatasetMaker(abc.ABC):
    config: DatasetMakingConfig
    
    def create(self):
        self.config.destination.mkdir(exist_ok=self.config.exists_ok, parents=True)
        
        with open(self.config.destination / f'{self.config.dataset_name}_coco_config.yaml', 'w') as f:
            pyrallis.dump(self.config, f, allow_unicode=False)
        
        text_category = {
                "id": 1,
                "name": "text",
                "supercategory": "beverage"
                }
        
        num_images = self.get_number_of_images()
        annot_counter = 0
        
        div_name_to_image_dicts = defaultdict(list)
        div_name_to_annot_dicts = defaultdict(list)
        
        for i in tqdm(range(num_images), desc=f'Processing {self.config.dataset_name}'):
            im, xywh_boxes, im_path, div_name, ignore_labels, texts = self[i]
            
            if len(xywh_boxes) == 0:
                warnings.warn(f"Skipping {im_path.name} at {im_path.parent} since it has no annotations")
                continue
            
            should_skip = False
            
            if texts is not None and self.config.space_policy != SpacePolicies.KEEP:
                for j, t in enumerate(texts):
                    if ' ' in t:
                        if self.config.space_policy == SpacePolicies.MARK_IGNORE:
                            ignore_labels[j] = True
                        elif self.config.space_policy == SpacePolicies.SKIP_IMAGE:
                            warnings.warn(f"Skipping {im_path.name} at {im_path.parent} since the {j} annotation contains space: `{t}`")
                            should_skip = True
                            break
                        else:
                            raise ValueError(f"Unsupported space policy {self.config.space_policy}")
            if should_skip:
                continue
            
            if self.config.ignore_policy == IgnorePolicies.KEEP:
                relevant_indexes = list(range(len(xywh_boxes)))
            elif self.config.ignore_policy == IgnorePolicies.REMOVE_ANNOT:
                relevant_indexes = [j for j, should_ignore in enumerate(ignore_labels) if not should_ignore]
            elif self.config.ignore_policy == IgnorePolicies.SKIP_IMAGE:
                
                relevant_indexes = [j for j, should_ignore in enumerate(ignore_labels) if not should_ignore]
                if len(xywh_boxes) > 0 and any(ignore_labels):
                    warnings.warn(f"Skipping {im_path.name} at {im_path.parent} since it has ignore annotations")
                    continue
            else:
                raise ValueError(f'Unsupported ignore policy {self.config.ignore_policy}')
            
            if len(relevant_indexes) == 0 and self.config.skip_image_without_gt_annots:
                warnings.warn(f"Skipping {im_path.name} at {im_path.parent} since it has no gt annotations")
                continue
            
            relevant_indexes = np.array(relevant_indexes).astype(int)
            xywh_boxes = np.array(xywh_boxes)[relevant_indexes]
            ignore_labels = np.array(ignore_labels)[relevant_indexes].astype(bool).astype(int)
            texts = np.array(texts)[relevant_indexes] if texts is not None else None
            
            areas = xywh_boxes[:, 2] * xywh_boxes[:, 3]
            
            file_name = f'{i}_' + '_'.join(str(im_path).split('/')[-3:])
            unique_im_id_string = f'{self.config.dataset_name}_{i}'
            im_id = int(hashlib.md5(unique_im_id_string.encode()).hexdigest()[:10], 16)
            
            cur_img_dict = {
                "id": im_id,
                "coco_url": "", 
                "flickr_url": "", 
                "width": im.width, 
                "height": im.height, 
                "file_name": file_name,
                "date_captured": "2024-06-10 10:00:00"
            }
            
            cur_img_annots = [
                {
                    "id": int(hashlib.md5(f"{self.config.dataset_name}_{annot_counter+j}".encode()).hexdigest()[:10], 16), 
                    "category_id": text_category['id'], 
                    "iscrowd": 0, 
                    "image_id": im_id, 
                    "area": float(areas[j]), 
                    "bbox": [list(map(float, xywh_boxes[j]))],
                    "ignore": int(bool(ignore_labels[j]))
                }

                for j in range(len(xywh_boxes))]
            
            if texts is not None:
                for j, t in enumerate(texts):
                    cur_img_annots[j]['text'] = str(t)
                    cur_img_annots[j]['ord'] = list(map(ord, t))
            
            annot_counter += len(xywh_boxes)
            
            div_name_to_image_dicts[div_name].append(cur_img_dict)
            div_name_to_annot_dicts[div_name].extend(cur_img_annots)
            img_save_dir = self.config.destination / f'{div_name}_images'
            img_save_dir.mkdir(exist_ok=True)
            im.save(img_save_dir / cur_img_dict["file_name"])

        for div_name in div_name_to_image_dicts.keys():
            coco_dict = {
                "info": {
                    "description": f"{self.config.dataset_name} {div_name} Dataset",
                    # "url": "https://github.com/persiandataset/Arshasb",
                    # "version": "1.0",
                    # "year": 2021,
                    # "contributor": "COCO Consortium",
                    # "date_created": "2021/11/24"
                    **self.config.metadata
                },
                "licenses": [],
                "images": div_name_to_image_dicts[div_name],
                "annotations": div_name_to_annot_dicts[div_name],
                "categories": [text_category]
            }
            with open(self.config.destination / f'{self.config.dataset_name}_{div_name}.json', 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, ensure_ascii=False)
            
    @abc.abstractmethod
    def get_number_of_images(self) -> int:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Image.Image, XYWH_BOXES, Path, str, List[bool], Optional[List[str]]]:
        raise NotImplementedError()
        
    

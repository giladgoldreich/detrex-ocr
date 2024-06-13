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
from PIL import Image, ImageDraw
from collections import defaultdict
import pyrallis
from tqdm.contrib.concurrent import thread_map

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
    test_run: bool = False
    max_workers: int = 8
    draw: bool = True

    def __post_init__(self):
        if self.add_dataset_name_to_destination:
            self.destination = self.destination / self.dataset_name
            self.add_dataset_name_to_destination = False


@dataclass
class CocoDatasetMaker(abc.ABC):
    config: DatasetMakingConfig
    text_category: Dict[str, Any] = field(default_factory=lambda:
                                          {
                                              "id": 1,
                                              "name": "text",
                                              "supercategory": "beverage"
                                          })

    def create(self):
        self.config.destination.mkdir(
            exist_ok=self.config.exists_ok, parents=True)

        with open(self.config.destination / f'{self.config.dataset_name}_coco_config.yaml', 'w') as f:
            pyrallis.dump(self.config, f, allow_unicode=False, sort_keys=False)

        num_images = self.get_number_of_images()
        if self.config.test_run:
            warnings.warn('Running in test run mode - only 10 images')
            num_images = min(num_images, 10)

        div_name_to_image_dicts = defaultdict(list)
        div_name_to_annot_dicts = defaultdict(list)

        all_results = thread_map(self.get_div_name_and_coco_annot_dict,
                                 list(range(num_images)),
                                 max_workers=max(self.config.max_workers, 1),
                                 disable=False,
                                 desc=f'Running on {self.config.dataset_name} with {self.config.max_workers} workers')
        
        all_im_ids = set()
        all_annot_ids = set()
        
        for i, (div_name, img_dict, img_annot_dict_list) in enumerate(tqdm(all_results, desc='merging')):
            if div_name is None:
                continue
            
            # making sure img id is unique in dataset
            assert img_dict['id'] not in all_im_ids
            all_im_ids.add(img_dict['id'])
            
            # updating annot ids and making sure that they are unique
            for annot_dict in img_annot_dict_list:
                assert annot_dict['id'] not in all_annot_ids
                all_annot_ids.add(annot_dict['id'])
            
            div_name_to_image_dicts[div_name].append(img_dict)
            div_name_to_annot_dicts[div_name].extend(img_annot_dict_list)


        for div_name in tqdm(div_name_to_image_dicts.keys(),
                             total=len(div_name_to_image_dicts.keys()),
                             desc='Saving jsons'):
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
                "categories": [self.text_category]
            }
            with open(self.config.destination / f'{self.config.dataset_name}_{div_name}.json', 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, ensure_ascii=False)

    @abc.abstractmethod
    def get_number_of_images(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Image.Image, XYWH_BOXES, str, str, List[bool], Optional[List[str]]]:
        raise NotImplementedError()

    def get_div_name_and_coco_annot_dict(self, im_num: int) -> Tuple[str, Dict, List[Dict]]:
        im, xywh_boxes, save_name, div_name, ignore_labels, texts = self[im_num]

        if im is None:
            return None, {}, []

        if len(xywh_boxes) == 0:
            warnings.warn(f"Skipping {save_name} since it has no annotations")
            return None, {}, []

        if texts is not None and self.config.space_policy != SpacePolicies.KEEP:
            for j, t in enumerate(texts):
                if ' ' in t:
                    if self.config.space_policy == SpacePolicies.MARK_IGNORE:
                        ignore_labels[j] = True
                    elif self.config.space_policy == SpacePolicies.SKIP_IMAGE:
                        warnings.warn(
                            f"Skipping {save_name} since the {j} annotation contains space: `{t}`")
                        return None, {}, []

                    else:
                        raise ValueError(
                            f"Unsupported space policy {self.config.space_policy}")

        if self.config.ignore_policy == IgnorePolicies.KEEP:
            relevant_indexes = list(range(len(xywh_boxes)))

        elif self.config.ignore_policy == IgnorePolicies.REMOVE_ANNOT:
            relevant_indexes = [j for j, should_ignore in enumerate(
                ignore_labels) if not should_ignore]

        elif self.config.ignore_policy == IgnorePolicies.SKIP_IMAGE:
            relevant_indexes = [j for j, should_ignore in enumerate(
                ignore_labels) if not should_ignore]
            if len(xywh_boxes) > 0 and any(ignore_labels):
                warnings.warn(
                    f"Skipping {save_name} since it has ignore annotations")
                return None, {}, []

        else:
            raise ValueError(
                f'Unsupported ignore policy {self.config.ignore_policy}')

        if len(relevant_indexes) == 0 and self.config.skip_image_without_gt_annots:
            warnings.warn(
                f"Skipping {save_name} since it has no gt annotations")
            return None, {}, []

        relevant_indexes = np.array(relevant_indexes).astype(int)
        xywh_boxes = np.array(xywh_boxes)[relevant_indexes]
        ignore_labels = np.array(ignore_labels)[
            relevant_indexes].astype(bool).astype(int)
        texts = np.array(texts)[
            relevant_indexes] if texts is not None else None

        areas = xywh_boxes[:, 2] * xywh_boxes[:, 3]

        file_name = f'{im_num}_{save_name}'
        # unique im id, should be unique within dataset
        unique_im_id_string = f'{self.config.dataset_name}_{im_num}'
        im_id = int(hashlib.md5(unique_im_id_string.encode()).hexdigest(), 16)

        cur_img_dict = {
            "id": im_id,
            "coco_url": "",
            "flickr_url": "",
            "width": im.width,
            "height": im.height,
            "file_name": file_name,
            "date_captured": "2024-06-10 10:00:00"
        }
        
        # annot ids are arbitrary and will be modified when merging the datasets

        cur_img_annots = [
            {
                "id": int(hashlib.md5(f'{unique_im_id_string}_{j}'.encode()).hexdigest(), 16),
                "category_id": self.text_category['id'],
                "iscrowd": 0,
                "image_id": im_id,
                "area": float(areas[j]),
                "bbox": list(map(float, xywh_boxes[j])),
                "ignore": int(bool(ignore_labels[j]))
            }

            for j in range(len(xywh_boxes))]

        if texts is not None:
            for j, t in enumerate(texts):
                cur_img_annots[j]['text'] = str(t)
                cur_img_annots[j]['ord'] = list(map(ord, t))

        img_save_dir = self.config.destination / f'{div_name}_images'
        img_save_dir.mkdir(exist_ok=True)
        im = im.convert('RGB')
        im.save(img_save_dir / cur_img_dict["file_name"])
        
        if self.config.draw:
            im_with_bboxes = im.copy()
            draw = ImageDraw.Draw(im_with_bboxes)
            for ann in cur_img_annots:
                draw.rectangle([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],
                               outline='red' if ann['ignore'] else 'green')
            bboxes_save_dir = Path(str(img_save_dir) + '_marked')
            bboxes_save_dir.mkdir(exist_ok=True)
            im_with_bboxes.save(Path(bboxes_save_dir / cur_img_dict['file_name']).with_suffix('.jpg'))
            

        return div_name, cur_img_dict, cur_img_annots

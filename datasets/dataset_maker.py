import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable, Tuple, List, Dict, Optional, Any, NamedTuple, Container
import abc
from tqdm import tqdm
import warnings
import numpy as np
import json
from PIL import Image, ImageDraw
from collections import defaultdict
import pyrallis
from tqdm.contrib.concurrent import thread_map
import shutil
import cv2
from detectron2.data.detection_utils import _apply_exif_orientation

class ImageWithAnnots(NamedTuple):
    im: Image.Image
    xy_coords: Container[Container[Tuple[float, float]]]
    ignore_mask: List[bool]
    subset: str
    save_name: Optional[str] = None
    original_img_path: Optional[Path] = None
    texts: Optional[List[str]] = None
    img_id: Optional[int] = None
    annot_ids: Optional[List[int]] = None
    
    @property
    def num_annots(self) -> int:
        return len(self.xy_coords)

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
    destination: Path = Path('/nfs/private/gilad/ocr_detection/detrex-ocr/datasets')
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
        if self.test_run:
            warnings.warn("setting `max_workers`=0 when in test run")
            self.max_workers = 0


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
        print(self.config.destination.absolute())

        with open(self.config.destination / f'{self.config.dataset_name}_coco_config.yaml', 'w') as f:
            pyrallis.dump(self.config, f, allow_unicode=False, sort_keys=False)

        num_images = self.get_number_of_images()
        if self.config.test_run:
            num_images = min(num_images, 10)
            warnings.warn(f'Running in test run mode - only {num_images} images')

        subset_to_image_dicts = defaultdict(list)
        subset_to_annot_dicts = defaultdict(list)
        
        if self.config.max_workers <= 1:
            
            all_results = []
            for i in tqdm(range(num_images), desc=f'Running on {self.config.dataset_name} with {self.config.max_workers} workers'):
                all_results.append(self.get_subset_and_coco_annot_dict(i))
        else:

            all_results = thread_map(self.get_subset_and_coco_annot_dict,
                                    list(range(num_images)),
                                    max_workers=self.config.max_workers,
                                    disable=False,
                                    desc=f'Running on {self.config.dataset_name} with {self.config.max_workers} workers')
        
        all_im_ids = set()
        all_annot_ids = set()
        all_file_names = set()
        
        for i, (subset, img_dict, img_annot_dict_list) in enumerate(tqdm(all_results, desc='merging')):
            if subset is None:
                continue
            
            # making sure img id is unique in dataset
            assert img_dict['id'] not in all_im_ids
            all_im_ids.add(img_dict['id'])
            
            assert img_dict['file_name'] not in all_file_names
            all_file_names.add(img_dict['file_name'])
            
            # updating annot ids and making sure that they are unique
            for annot_dict in img_annot_dict_list:
                assert annot_dict['id'] not in all_annot_ids
                all_annot_ids.add(annot_dict['id'])
            
            subset_to_image_dicts[subset].append(img_dict)
            subset_to_annot_dicts[subset].extend(img_annot_dict_list)


        for subset in tqdm(subset_to_image_dicts.keys(),
                             total=len(subset_to_image_dicts.keys()),
                             desc='Saving jsons'):
            coco_dict = {
                "info": {
                    "description": f"{self.config.dataset_name} {subset} Dataset",
                    # "url": "https://github.com/persiandataset/Arshasb",
                    # "version": "1.0",
                    # "year": 2021,
                    # "contributor": "COCO Consortium",
                    # "date_created": "2021/11/24"
                    **self.config.metadata
                },
                "licenses": [],
                "images": subset_to_image_dicts[subset],
                "annotations": subset_to_annot_dicts[subset],
                "categories": [self.text_category]
            }
            
            print(f"Num images in {subset}: {len(coco_dict['images'])}")
            print(f"Num annots in {subset}: {len(coco_dict['annotations'])}")

            
            with open(self.config.destination / f'{self.config.dataset_name}_{subset}.json', 'w', encoding='utf-8') as f:
                json.dump(coco_dict, f, ensure_ascii=False)
            
            
    @abc.abstractmethod
    def get_number_of_images(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Optional[ImageWithAnnots]:
        raise NotImplementedError()

    def get_subset_and_coco_annot_dict(self, im_num: int) -> Tuple[str, Dict, List[Dict]]:
        img_with_annots = self[im_num]
        
        if img_with_annots is None:
            return None, {}, []
        
        if img_with_annots.img_id is not None:
            img_id = img_with_annots.img_id
        else:
            unique_im_id_string = f'{self.config.dataset_name}_{img_with_annots.save_name if img_with_annots.save_name is not None else im_num}'
            img_id = int(hashlib.md5(unique_im_id_string.encode()).hexdigest()[:12], 16)
        
        if img_with_annots.annot_ids is not None:
            annot_ids = np.array(img_with_annots.annot_ids)
        else:
            annot_ids = np.array([int(hashlib.md5(f'{img_id}_{j}'.encode()).hexdigest()[:12], 16) for j in range(img_with_annots.num_annots)])
        
        if img_with_annots.save_name is not None:
            save_name = img_with_annots.save_name
        else:
            save_name = f'{img_id}.png'
            
        im = img_with_annots.im
        copy_original_image = img_with_annots.original_img_path is not None
        if hasattr(im, 'getexif'):
            fixed_im = _apply_exif_orientation(im)
            if fixed_im != im:
                warnings.warn(f"Changing image because its exif changes it: {save_name}")
                im = Image.fromarray(np.array(im))
                copy_original_image = False
            
        cur_img_dict = {
            "id": img_id,
            "coco_url": "",
            "flickr_url": "",
            "width": im.width,
            "height": im.height,
            "file_name": save_name,
            "date_captured": "2024-06-10 10:00:00"
        }
        
        if img_with_annots.num_annots == 0:
            warnings.warn(f"Skipping {save_name} since it has no annotations")
            return None, {}, []

        texts = img_with_annots.texts
        ignore_mask = img_with_annots.ignore_mask
        
        if texts is not None and self.config.space_policy != SpacePolicies.KEEP:
            for j, t in enumerate(texts):
                if ' ' in t:
                    if self.config.space_policy == SpacePolicies.MARK_IGNORE:
                        ignore_mask[j] = True
                    elif self.config.space_policy == SpacePolicies.SKIP_IMAGE:
                        warnings.warn(
                            f"Skipping {save_name} since the {j} annotation contains space: `{t}`")
                        return None, {}, []

                    else:
                        raise ValueError(f"Unsupported space policy {self.config.space_policy}")

        if self.config.ignore_policy == IgnorePolicies.KEEP:
            relevant_indexes = list(range(img_with_annots.num_annots))

        elif self.config.ignore_policy == IgnorePolicies.REMOVE_ANNOT:
            relevant_indexes = [j for j, should_ignore in enumerate(ignore_mask) if not should_ignore]

        elif self.config.ignore_policy == IgnorePolicies.SKIP_IMAGE:
            relevant_indexes = [j for j, should_ignore in enumerate(ignore_mask) if not should_ignore]
            if img_with_annots.num_annots > 0 and any(ignore_mask):
                # warnings.warn(f"Skipping {save_name} since it has ignore annotations")
                return None, {}, []

        else:
            raise ValueError(f'Unsupported ignore policy {self.config.ignore_policy}')

        if len(relevant_indexes) == 0 and self.config.skip_image_without_gt_annots:
            # warnings.warn(f"Skipping {save_name} since it has no gt annotations")
            return None, {}, []

        relevant_indexes = np.array(relevant_indexes).astype(int)
        relevant_indexes_set = set(relevant_indexes)
        xy_coords = [xy for j, xy in enumerate(img_with_annots.xy_coords) if j in relevant_indexes_set]
        xywh_boxes = self.xy_coords_to_xywh_boxes(xy_coords)
        ignore_mask = np.array(ignore_mask)[relevant_indexes].astype(bool).astype(int)
        texts = np.array(texts)[relevant_indexes] if texts is not None else None
        annot_ids = np.array(annot_ids)[relevant_indexes]
        areas = xywh_boxes[:, 2] * xywh_boxes[:, 3]
            
        cur_img_annots = [
            {
                "id": int(annot_ids[j]),
                "category_id": self.text_category['id'],
                "iscrowd": int(bool(ignore_mask[j])),
                "image_id": img_id,
                "area": float(areas[j]),
                "bbox": list(map(float, xywh_boxes[j])),
                "segmentation": [
                    list(map(float, np.array(xy_coords[j]).flatten().astype(float)))
                ],
                "ignore": int(bool(ignore_mask[j]))
            }

            for j in range(len(xywh_boxes))]

        if texts is not None:
            for j, t in enumerate(texts):
                cur_img_annots[j]['text'] = str(t)
                cur_img_annots[j]['ord'] = list(map(ord, t))
                
        img_save_dir = self.config.destination / f'{img_with_annots.subset}_images'
        img_save_dir.mkdir(exist_ok=True)
        img_full_save_path = img_save_dir / save_name
        if img_with_annots.original_img_path is not None and copy_original_image:
            shutil.copy(src=img_with_annots.original_img_path, dst=img_full_save_path)
        else:
            im.convert('RGB').save(img_full_save_path)
                    
        if self.config.draw:
            im_with_bboxes = im.copy().convert('RGB')
            draw = ImageDraw.Draw(im_with_bboxes)
            for ann in cur_img_annots:
                for poly in ann['segmentation']:
                    draw.polygon(list(map(int, poly)),
                                 outline='red' if ann['ignore'] else 'green')
                    
                # draw.rectangle([ann['bbox'][0],
                #                     ann['bbox'][1],
                #                     ann['bbox'][0] + ann['bbox'][2],
                #                     ann['bbox'][1] + ann['bbox'][3]],
                #                     outline='red' if ann['ignore'] else 'green')
                    
            bboxes_save_dir = Path(str(img_save_dir) + '_marked')
            bboxes_save_dir.mkdir(exist_ok=True)
            im_with_bboxes.save(Path(bboxes_save_dir / save_name).with_suffix('.jpg'))            

        return img_with_annots.subset, cur_img_dict, cur_img_annots

    @staticmethod
    def xy_coords_to_xywh_boxes(xy_coords: Container[Container[Tuple[float, float]]]) -> Container[Tuple[float, float, float, float]]:
        if len(xy_coords) == 0:
            return np.zeros((0, 4), dtype=float)
        xywh_boxes = []
        for cur_xyxy in xy_coords:
            cur_xyxy_arr = np.array(cur_xyxy).astype(float).reshape(-1, 2)
            xywh_boxes.append([
                cur_xyxy[:, 0].min(),
                cur_xyxy[:, 1].min(),
                cur_xyxy[:, 0].max() - cur_xyxy[:, 0].min(),
                cur_xyxy[:, 1].max() - cur_xyxy[:, 1].min(),
            ])
        xywh_boxes = np.array(xywh_boxes).astype(float)
        return xywh_boxes
    
    @staticmethod
    def xy_coords_to_cxcywha_boxes(xy_coords: Container[Container[Tuple[float, float]]]) -> Container[Tuple[float, float, float, float, float]]:
        if len(xy_coords) == 0:
            return np.zeros((0, 5), dtype=float)
        cxcywha_boxes = [cv2.minAreaRect(np.array(xyxy).reshape(-1, 2).astype(int)) for xyxy in xy_coords]
        cx = np.array([t[0][0] for t in cxcywha_boxes])
        cy = np.array([t[0][1] for t in cxcywha_boxes])
        w = np.array([t[1][0] for t in cxcywha_boxes])
        h = np.array([t[1][1] for t in cxcywha_boxes])
        alpha = np.array([t[2] for t in cxcywha_boxes])
        return np.stack([cx, cy, w, h, alpha], axis=-1).astype(float)
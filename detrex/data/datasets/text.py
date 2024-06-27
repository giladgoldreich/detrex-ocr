# Taken & modified from AdelaiDet repo
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import hashlib
import io
import logging
import os
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse COCO-format text annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger("detectron2")

__all__ = ["load_text_json", "register_text_instances"]


def register_text_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in json annotation format for text detection and recognition.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(
        name, lambda: load_text_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="text", **metadata
    )


def load_text_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, rotated_boxes: bool = False, fix_object_ids: bool = True):
    """
    Load a json file with totaltext annotation format.
    Currently supports text detection and recognition.

    Args:
        json_file (str): full path to the json file in totaltext annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
        
        # additions
        
        rotated_boxes (bool): whether to convert the boxes to rotated boxes
        fix_object_ids (bool): whether to make object ids unique (helps when there is more than one dataset and you want to make sure that the object ids will be unique)

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    logger.info(f'Starting to load dataset {dataset_name} from {json_file}')
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"]
                         for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'rec': [84, 72, ... 96],
    #   'bezier_pts': [169.0, 425.0, ..., ]
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"]
                   for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "rec", "category_id"] + \
        (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    annot_counter = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        
        # adding dataset name to the record, makes everything else much easier (sampling, evaluating, etc.)
        record['dataset_name'] = dataset_name

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            # we do allow ignore here
            # assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(
                        poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm
            
            # adding keypoints - not original repo
            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            bezierpts = anno.get("bezier_pts", None)
            # Bezier Points are the control points for BezierAlign Text recognition (BAText)
            if bezierpts:  # list[float]
                obj["beziers"] = bezierpts

            text = anno.get("rec", None)
            if text:
                obj["text"] = text
                
            # Rotated bboxes support    
            # Deciding on box mode, based on rotated_boxes
            if rotated_boxes:
                # If we have the rotated box, we work with it
                if anno.get('rotated_box'):
                    obj['bbox'] = rotated_box_anno_to_xywha(anno['rotated_box'])
                else:
                    obj['bbox'] = BoxMode.convert(obj['bbox'], from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYWHA_ABS)
                obj['bbox_mode'] = BoxMode.XYWHA_ABS
            else:
                obj['bbox_mode'] = BoxMode.XYWH_ABS

            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
                
            # Setting a unique id for each annotation box (CocoDatasetCreator sets unique ids only within each dataset)
            
            if fix_object_ids:
                assert dataset_name is not None, "Cannot fix object ids when dataset name is missing"
                obj['origin_id'] = obj.get('id')
                unique_id_string = '{}_{}'.format(dataset_name, obj.get('id', annot_counter))
                obj['id'] = int(hashlib.md5(unique_id_string.encode()).hexdigest(), 16)
                
            objs.append(obj)
            annot_counter += 1
        
        
        record["annotations"] = objs
        
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def rotated_box_anno_to_xywha(rotated_box):
    np_box = np.array(rotated_box)  # An array with 4x2 vertices, the first is top left, going clockwise
    center_x, center_y = np.mean(np_box, axis=0)
    # width is measured on the first side of np_box
    width = np.linalg.norm(np_box[1] - np_box[0])
    # width is measured on the second side of np_box
    height = np.linalg.norm(np_box[2] - np_box[1])
    # Angle is measures as the tangens of the upper side vs its x,y values
    angle = np.rad2deg(np.arctan2(np_box[0, 1] - np_box[1, 1], np_box[1, 0] - np_box[0, 0]))
    return [center_x, center_y, width, height, angle]
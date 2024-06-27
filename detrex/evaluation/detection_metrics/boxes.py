# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from detectron2.structures.boxes import Boxes
import numpy as np
from typing import List, Tuple, Union
import torch
import math

from detectron2.structures.boxes import BoxMode as D2BoxMode
from detectron2.structures.boxes import Boxes as D2Boxes
from detectron2.structures.boxes import pairwise_ioa as d2_pairwise_ioa
from detectron2.structures.boxes import pairwise_intersection as d2_pairwise_intersection
from detectron2.structures.rotated_boxes import pairwise_iou_rotated as d2_pairwise_iou_rotated
from detectron2.structures.rotated_boxes import RotatedBoxes

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

# Making sure we have the correct classes loaded from the original boxes model
BoxMode = D2BoxMode
Boxes = D2Boxes
pairwise_ioa = d2_pairwise_ioa
pairwise_intersection = d2_pairwise_intersection

"""
@tsiper - custom memory optimized implementations of the pairwise_iou method which overrides the original
"""


def pairwise_iou(boxes1, boxes2, interval: int = 50, inter_over_min_area=False) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1 (Boxes): Contains N boxes
        boxes2 (Boxes): Contains M boxes
        interval (int): The size interval in which we skip

    Returns:
        Tensor: IoU, sized [N,M].
    """
    boxes1_tensor = boxes1 if isinstance(boxes1, torch.Tensor) else boxes1.tensor
    boxes2_tensor = boxes2 if isinstance(boxes2, torch.Tensor) else boxes2.tensor

    # In case we deal with rotated boxes
    if boxes1_tensor.shape[1] == 5 and boxes2_tensor.shape[1] == 5:
        return pairwise_iou_rotated(boxes1=boxes1, boxes2=boxes2, inter_over_min_area=inter_over_min_area)

    area1 = (boxes1_tensor[:, 2] - boxes1_tensor[:, 0]) * (boxes1_tensor[:, 3] - boxes1_tensor[:, 1])
    area2 = (boxes2_tensor[:, 2] - boxes2_tensor[:, 0]) * (boxes2_tensor[:, 3] - boxes2_tensor[:, 1])
    # tsiper: here's the main modification - batching over the first box dimension
    iou = torch.zeros(0, boxes2_tensor.shape[0]).to(boxes1_tensor.device)  # A tensor with no dimension
    for i in range((boxes1_tensor.shape[0] - 1) // interval + 1):
        boxes1_tensor_batch = boxes1_tensor[i * interval:(i + 1) * interval, :]  # [N//interval,4]
        area1_batch = area1[i * interval: (i + 1) * interval]
        width_height = torch.min(boxes1_tensor_batch[:, None, 2:], boxes2_tensor[:, 2:]) - torch.max(
            boxes1_tensor_batch[:, None, :2], boxes2_tensor[:, :2])  # [N//interval,M,2]
        width_height.clamp_(min=0)  # [N//interval,M,2]
        inter = width_height.prod(dim=2)  # [N//interval,M]  # The intersection
        del width_height
        if inter_over_min_area:
            max_area = torch.min(area1_batch[:, None], area2)
            iou = torch.cat((iou, torch.where(inter > 0, inter / max_area,
                                              torch.zeros(1, dtype=inter.dtype, device=inter.device),
                                              )), dim=0)
        else:
            iou = torch.cat((iou, torch.where(inter > 0, inter / (area1_batch[:, None] + area2 - inter),
                                              torch.zeros(1, dtype=inter.dtype, device=inter.device),
                                              )), dim=0)
        del inter
    return iou


def pairwise_iou_rotated(boxes1: RotatedBoxes, boxes2: RotatedBoxes, interval: int = 50,
                         inter_over_min_area=False) -> torch.Tensor:
    # Making sure we operate on the tensors and note the RotatedBoxes
    boxes1_tensor = boxes1 if isinstance(boxes1, torch.Tensor) else boxes1.tensor
    boxes2_tensor = boxes2 if isinstance(boxes2, torch.Tensor) else boxes2.tensor

    intersection_func = _pairwise_ioa_rotated if inter_over_min_area else d2_pairwise_iou_rotated
    iou = torch.zeros(0, boxes2_tensor.shape[0]).to(boxes1_tensor.device)  # A tensor with no dimension
    for i in range((boxes1_tensor.shape[0] - 1) // interval + 1):
        iou_batch = intersection_func(boxes1_tensor[i * interval:(i + 1) * interval, :], boxes2_tensor)
        iou = torch.cat((iou, iou_batch), dim=0)
    return iou


def _pairwise_ioa_rotated(boxes1_tensor: torch.Tensor, boxes2_tensor: torch.Tensor):
    """
    Computes the intersection over minimal area for rotated boxes
    :param boxes1_tensor: an M x 5 tensor describing rotated boxes in absolute coordinates
    :param boxes2_tensor: an N x 5 tensor describing rotated boxes in absolute coordinates
    :return: An Intersection-Over-Min-Area tensor (M x N)
    """
    assert (boxes1_tensor.shape[1] == 5) and (boxes2_tensor.shape[1] == 5), "Input tensors don't describe rotated boxes"

    # Using the d2 C based method for fast computation of IoU
    iou = d2_pairwise_iou_rotated(boxes1_tensor, boxes2_tensor)  # M x N

    # We compose matrices of the areas, for mesh computations
    area1 = boxes1_tensor[:, 2] * boxes1_tensor[:, 3]  # M x 1
    area2 = boxes2_tensor[:, 2] * boxes2_tensor[:, 3]  # N x 1
    area1_mesh = area1.repeat(len(boxes2_tensor), 1).T  # M x N area mesh
    area2_mesh = area2.repeat(len(boxes1_tensor), 1)  # M x N area mesh

    # By definition IoU = Intersection / (Area1 + Area2 - Intersection)
    # Therefore we isolate Intersection by "Intersection = IoU * (Area1 + Area2) / (1 + IoU)"
    intersection = (area1_mesh + area2_mesh) * iou / (1 + iou)

    # Now we divide by the minimal area to obtain the intersection over min area metric
    ioa = intersection / torch.min(area1_mesh, area2_mesh)

    return ioa

def box_to_rbox(box_tensor):
    arr = BoxMode.convert(box_tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    original_dtype = arr.dtype
    arr = arr.double()
    arr[:, 0] += arr[:, 2] / 2.0
    arr[:, 1] += arr[:, 3] / 2.0
    angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype, device=arr.device)
    arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
    return arr


def rbox_to_box(rbox_tensor):
    ret = BoxMode.convert(rbox_tensor, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS)
    return ret


class MyBoxMode:
    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        one_dim_flag = False
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
                one_dim_flag = len(arr.shape) == 1
            else:
                arr = box.clone()

        assert to_mode.value not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ] and from_mode.value not in [
                   BoxMode.XYXY_REL,
                   BoxMode.XYWH_REL,
               ], "Relative mode not yet supported!"

        if one_dim_flag:
            # Add another dimension to arr
            arr = arr.unsqueeze(0)

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                    arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"

            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)

        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if one_dim_flag:
            # remove extra dimension
            arr = arr[0]

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr

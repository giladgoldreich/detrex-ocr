from detectron2.data.samplers import RepeatFactorTrainingSampler
from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import torch
# from loguru import logger
import logging
from detectron2.data.build import (filter_images_with_only_crowd_annotations, torchdata, 
                                   itertools, filter_images_with_few_keypoints,
                                   check_metadata_consistency, print_instances_class_histogram, 
                                   DatasetCatalog, load_proposals_into_dataset, MetadataCatalog, build_detection_train_loader)
from detectron2.data.samplers import RepeatFactorTrainingSampler, TrainingSampler


def get_dataset_dicts_and_sampler(
    names: Union[str, List[str]],
    weights: Optional[Union[float, List[float]]] = None,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
    seed: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[List[Dict], torchdata.Sampler]:
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        weights: (float or list[float] or None): weights for each dataset name. 
                  Single value (e.g. .2) means uniform sampling between datasets.
                  No values means no weighting. 
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.
        shuffle (bool): whether to shuffle the indices or not
        seed (int): the initial seed of the shuffle. Must be the same
            across all workers. If None, will use a random seed shared
            among workers (require synchronization among all workers).

    Returns:
        A tuple of two elements:
            1. list[dict]: a list of dicts following the standard dataset dict format.
            2. Sampler for the dicts
    """
    logger = logging.getLogger("detectron2")
    if isinstance(names, str):
        names = [names]

    assert len(names), names

    if weights is not None:
        if isinstance(weights, (int, float)):
            weights = [weights] * len(weights)

        weights = np.array(weights).astype(float).flatten()
        assert (weights != 0).any(), "All weights are 0"
        if len(weights) == 1:
            weights = len(names) * weights.tolist()
            weights = np.array(weights).astype(float)
        weights = weights / weights.sum()

        assert len(weights) == len(names), (weights, names)

    dataset_dicts = [DatasetCatalog.get(dataset_name)
                     for dataset_name in names]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            concat_dataset = torchdata.ConcatDataset(dataset_dicts)
        else:
            concat_dataset = dataset_dicts[0]

    else:
        if proposal_files is not None:
            assert len(names) == len(proposal_files)
            # load precomputed proposals from proposal files
            dataset_dicts = [
                load_proposals_into_dataset(dataset_i_dicts, proposal_file)
                for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
            ]

        has_instances = "annotations" in dataset_dicts[0][0]
        if filter_empty and has_instances:
            dataset_dicts = [filter_images_with_only_crowd_annotations(
                d) for d in dataset_dicts]
        if min_keypoints > 0 and has_instances:
            dataset_dicts = [filter_images_with_few_keypoints(
                d) for d in dataset_dicts]

        if check_consistency and has_instances:
            try:
                class_names = MetadataCatalog.get(names[0]).thing_classes
                check_metadata_consistency("thing_classes", names)
                print_instances_class_histogram(dataset_dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

        concat_dataset = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(concat_dataset), "No valid data found in {}.".format(
        ",".join(names))

    if len(dataset_dicts) == 1:
        logger.info("Single dataset -> naive sampling")
        weights = np.array([1.])
        sampler = TrainingSampler(len(concat_dataset))

    elif weights is not None:
        repeat_factors = list(itertools.chain.from_iterable(
            [[w] * len(ds) for w, ds in zip(weights, dataset_dicts)]))
        repeat_factors = np.array(repeat_factors).astype(float)
        repeat_factors = repeat_factors / repeat_factors.sum()
        repeat_factors = len(concat_dataset) * repeat_factors
        sampler = RepeatFactorTrainingSampler(
            repeat_factors=torch.from_numpy(repeat_factors), seed=seed, shuffle=shuffle)

    else:
        logger.info("No weights were given -> naive sampling")
        weights = np.array([len(d) for d in dataset_dicts]).astype(float)
        weights = weights / weights.sum()
        sampler = TrainingSampler(
            len(concat_dataset), shuffle=shuffle, seed=seed)

    for dataset_name, dataset, w in zip(names, dataset_dicts, weights):
        logger.info(
            f'{dataset_name} contains {len(dataset)} samples, weight: {w:3f}')

    return concat_dataset, sampler


def build_weighted_detection_train_loader(
    dataset,
    *,
    mapper,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset or tuple of (list/Dataset with sampler)): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable), with an optional sampler. 
            It can be obtained by using :func:`get_dataset_dicts_and_sampler` or :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
            The reason it is called dataset (and not dataset_and_sampler) is for backward compatability.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper`
    """
    if isinstance(dataset, tuple) and len(dataset) == 2 and isinstance(dataset[1], torchdata.Sampler):
        ds = dataset[0]
        sampler = dataset[1]
    else:
        ds = dataset
        sampler = None
    return build_detection_train_loader(
        dataset=ds,
        sampler=sampler,
        mapper=mapper,
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

import os
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy
from .matcher import Matcher
from .boxes import pairwise_iou, pairwise_iou_rotated
# from RekognitionHieroDetectron2.utils.iou_utils import tight_iou


class Metrics(ABC):

    def __init__(self, cfg=None, iou_threshold=0.5, default_score=0.4, match_algorithm='fast', name='',
                 iou_type='iou', len_scores_vec=51, save_instances=False, **kwargs):

        # The matcher is the algorithm that matches between predictions and ground-truths
        self.cfg = cfg
        self.iou_threshold = iou_threshold
        self.default_score = default_score
        self.save_instances = save_instances
        # We always define a matching algorithm for any metric subclass, so the matching can be done the same
        match_algorithm_dict = {
            'greedy': Matcher.match_greedy,
            'fast': Matcher.match_fast,
            'hungarian': Matcher.match_hungarian,
        }
        if match_algorithm not in match_algorithm_dict:
            raise Exception(f'Please choose a match algorithm from {match_algorithm_dict.keys()}')
        self.match_func = match_algorithm_dict[match_algorithm]
        self.dataframe = pd.DataFrame()
        self.metric_name = name

        # Defines the IoU computation function, the input to this function must be iou_func(boxes1, boxes2)
        # and the output is a matrix of N x M size with the affinity scores between the boxes lists
        iou_func_dict = {'iou': pairwise_iou,
                        #  'tight_iou': tight_iou,
                         'rotated_iou': pairwise_iou_rotated}
        self.iou_type = iou_type
        self.iou_func = iou_func_dict[self.iou_type]

        # We define a vector of scores, we calculate the metrics for all these score thresholds together
        self._scores_vec = np.linspace(0, 1, len_scores_vec)  # [0, 0.02, 0.4, ... , 1]
        self._default_score_idx = np.argmin(np.abs(self._scores_vec - self.default_score))

        # Whether we wish to save instances or not for visualization and other
        self.save_instances = save_instances

    @abstractmethod
    def add_entry(self, dataset_name, file_path, pred_instances, gt_instances):
        """
        This is the main method used for aggregating additional documents
        :param str dataset_name: The dataset name
        :param str file_path: The path to the file (we actually only save the file_name)
        :param Instances pred_instances: The prediction instances (N Boxes) for a given class (word/line/etc...)
        :param Instances gt_instances: The ground-truth instances for a given class (M Boxes)
        Updates self.dataframe to reflect the information for the new document
        """
        return NotImplemented

    def by_dataset(self, dataset_name):
        # Returns the OCR metrics for only a single dataset, useful for slicing and displaying results individually
        filtered_obj = self.filter_by_dataset(dataset_name)
        return filtered_obj.compute()
        # if dataset_name.lower() == 'all':
        #     return self.compute()
        # else:
        #     obj = deepcopy(self)
        #     obj.dataframe = obj.dataframe[obj.dataframe.dataset_name == dataset_name]
        #     return obj.compute()
        
    def filter_by_dataset(self, dataset_name):
        if dataset_name.lower() == 'all':
            return self
        else:
            obj = deepcopy(self)
            obj.dataframe = obj.dataframe[obj.dataframe.dataset_name == dataset_name]
            return obj

    def by_asset_id(self, asset_id):
        # Returns the OCR metrics for only a single dataset, useful for slicing and displaying results individually
        obj = deepcopy(self)
        obj.dataframe = obj.dataframe[obj.dataframe.asset_id == asset_id]
        return obj.compute()

    @property
    def dataset_names(self):
        return list(self.dataframe.dataset_name.unique())

    @property
    def num_assets(self):
        return len(self.dataframe.asset_id.unique())

    @abstractmethod
    def reset(self):
        """
        This method resets the internal properties used for the metric computation
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """
        Computes all of the metrics according to all of the aggregated documents
        :return: self - an OCRMetrics object, yet now with all the metric fields populated
        """
        raise NotImplementedError

    @abstractmethod
    def summary_dict(self):
        """
        Produces a summary dictionary with a lot of evaluation information.
        :return: A summary dict with all the OCR localization metrics with the following structure
        {
            dataset_name_1 : {
                metric_1: metric1_value
                metric_2: metric_value
            }
            dataset_name_2: { ... }
            ...
        }
        """
        return NotImplemented

    def summary_table(self, metric_names=None):
        """
        Returns a table
        :return:
        """
        summary_dict = self.summary_dict()
        # If no metric names are supplied we take all of the keys of the first inner dict
        metric_names = metric_names or list()
        summary_table = list()
        # Adding the table row
        summary_table.append(['dataset_name'] + metric_names)

        # Adding a row for each dataset
        for dataset_name, metrics_dict in sorted(summary_dict.items()):
            row = [dataset_name] + [metrics_dict[metric_name] for metric_name in metric_names]
            summary_table.append(row)

        return summary_table

    @abstractmethod
    def produce_figure_dict(self):
        """
        :return: A dict of figures to be visualized, i.e:
            {
                'PR_Curve': <pyplot.figure_object>,
                'Confusion Matrix: <seaborn.figure_object>,
            }
        """
        return NotImplemented

    def _idx_from_score(self, score=None):
        return self._default_score_idx if score is None else np.argmin(np.abs(self._scores_vec - score))

    def _update_dataframe(self, data, columns, dataset_name, file_path, pred_instances=None, gt_instances=None):
        asset_df = pd.DataFrame(data, columns=columns)
        asset_df['dataset_name'] = dataset_name
        asset_df['file_path'] = file_path
        asset_df['asset_id'] = os.path.splitext(os.path.basename(file_path))[0]

        asset_df['pred_instances'] = None
        asset_df.at[0, 'pred_instances'] = pred_instances

        asset_df['gt_instances'] = None
        asset_df.at[0, 'gt_instances'] = gt_instances

        self.dataframe = pd.concat([self.dataframe, asset_df])

    def __add__(self, other):
        new_obj = deepcopy(self)
        new_obj.dataframe = pd.concat([self.dataframe, other.dataframe])
        return new_obj

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

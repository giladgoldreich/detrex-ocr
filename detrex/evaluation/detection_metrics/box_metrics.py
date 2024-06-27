import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from detectron2.structures import Boxes, RotatedBoxes
from .metrics import Metrics
from .boxes import rbox_to_box


class BoxMetrics(Metrics):

    def __init__(self, *args, box_class=0, box_classes=None, save_instances=False, **kwargs):

        # The matcher is the algorithm that matches between predictions and ground-truths
        super().__init__(*args, **kwargs)

        # We define a vector of scores, we calculate the metrics for all these score thresholds together
        self.box_classes = box_classes

        if box_classes is None:
            self.box_classes = [box_class]

        self.save_instances = save_instances

        # The computed metrics, are populated using the self.compute() method
        self.accuracy_vec, self.precision_vec, self.recall_vec, self.fscore_vec = [None] * 4

    def add_entry(self, dataset_name, file_path, pred_instances, gt_instances):
        """
        This is the main method used for aggregating additional documents
        :param str dataset_name: The dataset name
        :param str file_path: The path to the file (we actually only save the file_name)
        :param Instances pred_instances: The prediction instances (N Boxes) for a given class (word/line/etc...)
        :param Instances gt_instances: The ground-truth instances for a given class (M Boxes)
        """
        # Extracting the prediction and gt boxes for the word class
        pred_boxes = []
        pred_scores = []
        gt_boxes = []

        # Evaluating together only the desired box_classes
        for box_class in self.box_classes:
            pred_boxes.append(pred_instances[pred_instances.pred_classes == box_class].pred_boxes)
            pred_scores.append(pred_instances[pred_instances.pred_classes == box_class].scores)
            gt_boxes.append(gt_instances[gt_instances.gt_classes == box_class].gt_boxes)

        if self.iou_type != 'rotated_iou':
            if isinstance(pred_boxes[0], RotatedBoxes):
                pred_boxes = [Boxes(rbox_to_box(b.tensor)) for b in pred_boxes]
            if isinstance(gt_boxes[0], RotatedBoxes):
                gt_boxes = [Boxes(rbox_to_box(b.tensor)) for b in gt_boxes]
        pred_boxes = pred_boxes[0].cat(pred_boxes)
        gt_boxes = gt_boxes[0].cat(gt_boxes)
        pred_scores = torch.cat(pred_scores)

        # Evaluating the extracted boxes, getting the number of true_positives, false_positives and fale_negatives
        tp, fp, fn = self._evaluate_boxes(pred_boxes, gt_boxes, pred_scores)
        data_columns = ['tp', 'fp', 'fn', 'score']
        data_tuples = list(zip(tp, fp, fn, self._scores_vec))

        # Building a dataframe for this asset and updating the main dataframe
        if self.save_instances:
            self._update_dataframe(data=data_tuples, columns=data_columns, dataset_name=dataset_name,
                                   file_path=file_path, gt_instances=gt_instances, pred_instances=pred_instances)
        else:
            self._update_dataframe(data=data_tuples, columns=data_columns, dataset_name=dataset_name,
                                   file_path=file_path)

        # We delete the metric scores after adding a document, since they're not updated
        self.reset()

    def compute(self):
        """
        Computes all of the metrics according to all of the aggregated documents
        :return: self - an OCRMetrics object, yet now with all the metric fields populated
        """

        sum_df = self.dataframe.groupby('score').sum(numeric_only=True)
        accuracy = sum_df.tp / (sum_df.tp + sum_df.fp + sum_df.fn)
        precision = sum_df.tp / (sum_df.tp + sum_df.fp)
        recall = sum_df.tp / (sum_df.tp + sum_df.fn)
        fscore = 2 * sum_df.tp / (2 * sum_df.tp + sum_df.fp + sum_df.fn)
        precision[~np.isfinite(precision)] = 1
        accuracy[~np.isfinite(accuracy)] = 0
        recall[~np.isfinite(recall)] = 0
        fscore[~np.isfinite(fscore)] = 0
        self.accuracy_vec, self.precision_vec, self.recall_vec, self.fscore_vec = accuracy.values, precision.values, recall.values, fscore.values

        return self

    def summary_df(self):
        """
        :return: Produces a dataframe with the result metrics
        """
        if any([x is None for x in [self.accuracy_vec, self.recall_vec, self.precision_vec, self.fscore_vec]]):
            self.compute()
        summary = pd.DataFrame(data=dict(
            scores=self._scores_vec,
            accuracy=self.accuracy_vec,
            precision=self.precision_vec,
            recall=self.recall_vec,
            fscore=self.fscore_vec,
        ))
        return summary

    def summary_dict(self):
        """
        Produces a summary dictionary with a lot of evaluation information.
        :return: A summary dict with all the OCR localization metrics
        """
        if any(x is None for x in [self.accuracy_vec, self.recall_vec, self.precision_vec, self.fscore_vec]):
            self.compute()
        summary = dict()
        # Getting the dataset names, adding the dataset keyword 'all' if there is more than one dataset
        dataset_names = self.dataset_names
        dataset_names = ['ALL'] + dataset_names if len(dataset_names) > 1 else dataset_names
        # Iterating over the datasets, computing the metrics for each
        for dataset_name in dataset_names:
            dataset_metrics = self.by_dataset(dataset_name)
            summary[dataset_name] = {
                'precision': dataset_metrics.precision,
                'recall': dataset_metrics.recall,
                'fscore': dataset_metrics.fscore,
                'optimal_precision': dataset_metrics.optimal_precision,
                'optimal_recall': dataset_metrics.optimal_recall,
                'optimal_threshold': dataset_metrics.optimal_score_threshold,
                'max_fscore': dataset_metrics.max_fscore,
                'num_docs': dataset_metrics.num_assets
            }
        return summary

    def summary_table(self, metric_names=None):
        metric_names = metric_names or ['num_docs', 'recall', 'precision', 'fscore', 'optimal_threshold', 'optimal_recall',
                                        'optimal_precision', 'max_fscore']
        return super().summary_table(metric_names)

    @property
    def optimal_score_threshold(self):
        if self.fscore_vec is None:
            self.compute()
        return self._scores_vec[np.argmax(self.fscore_vec)]

    @property
    def max_fscore(self):
        if self.fscore_vec is None:
            self.compute()
        return np.max(self.fscore_vec)

    @property
    def optimal_recall(self):
        if self.fscore_vec is None:
            self.compute()
        return self.recall_vec[np.argmax(self.fscore_vec)]

    @property
    def optimal_precision(self):
        if self.fscore_vec is None:
            self.compute()
        return self.precision_vec[np.argmax(self.fscore_vec)]

    def reset(self):
        self.accuracy_vec, self.precision_vec, self.recall_vec, self.fscore_vec = None, None, None, None

    def _evaluate_boxes(self, pred_boxes: Boxes, gt_boxes: Boxes, pred_scores: torch.Tensor, **kwargs):
        iou_mtx = self.iou_func(gt_boxes, pred_boxes)
        tp, fp, fn = list(), list(), list()
        for score in self._scores_vec:
            filtered_iou_mtx = iou_mtx[:, pred_scores > score]
            if filtered_iou_mtx.shape[1] == 0:
                num_matches = 0
            else:
                matches = self.match_func(filtered_iou_mtx, self.iou_threshold)
                if len(matches) > 0:
                    num_matches = len(matches[:, 1].unique())
                else:
                    num_matches = 0
            tp.append(num_matches)
            fp.append(filtered_iou_mtx.shape[1] - num_matches)
            fn.append(len(gt_boxes) - num_matches)
        return tp, fp, fn

    def get_accuracy(self, score=None):
        if self.accuracy_vec is None:
            self.compute()
        return self.accuracy_vec[self._idx_from_score(score)]

    accuracy = property(get_accuracy)

    def get_recall(self, score=None):
        if self.recall_vec is None:
            self.compute()
        return self.recall_vec[self._idx_from_score(score)]

    recall = property(get_recall)

    def get_precision(self, score=None):
        if self.precision_vec is None:
            self.compute()
        return self.precision_vec[self._idx_from_score(score)]

    precision = property(get_precision)

    def get_fscore(self, score=None):
        if self.fscore_vec is None:
            self.compute()
        return self.fscore_vec[self._idx_from_score(score)]

    fscore = property(get_fscore)

    def produce_figure_dict(self):
        figures_dict = {}
        summary_dict = self.summary_dict()
        for dataset_name in summary_dict.keys():
            filtered_obj = self.filter_by_dataset(dataset_name)
            df = filtered_obj.summary_df()
            # sns.set(style='darkgrid')

            # Plotting the main metrics in linear scale
            columns = ['fscore', 'recall', 'precision']
            metric_fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            for y in columns:
                sns.lineplot(data=df, x='scores', y=y, ax=axes[0])
            axes[0].legend(columns)

            # Plotting the main metrics in log scale, computing 1-y
            for y in columns:
                df[y + '_err'] = 1 - df[y]
                sns.lineplot(data=df, x='scores', y=y + '_err', ax=axes[1])
            axes[1].set(yscale='log')
            axes[1].legend([f'1 - {col}' for col in columns])  # These are metric errors
            for ax in axes:
                ax.set_ylabel('')
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
            metric_fig.suptitle(f'{self.metric_name}_Metrics/{dataset_name}')
            figures_dict[f'{self.metric_name}_Metrics/{dataset_name}'] = metric_fig
            
        return figures_dict
        
        ## gilad - removed old code
        # df = self.summary_df()
        # sns.set(style='darkgrid')

        # # Plotting the main metrics in linear scale
        # columns = ['fscore', 'recall', 'precision']
        # metric_fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        # for y in columns:
        #     sns.lineplot(data=df, x='scores', y=y, ax=axes[0])
        # axes[0].legend(columns)

        # # Plotting the main metrics in log scale, computing 1-y
        # for y in columns:
        #     df[y + '_err'] = 1 - df[y]
        #     sns.lineplot(data=df, x='scores', y=y + '_err', ax=axes[1])
        # axes[1].set(yscale='log')
        # axes[1].legend([f'1 - {col}' for col in columns])  # These are metric errors
        # for ax in axes:
        #     ax.set_ylabel('')

        # return {
        #     f'{self.metric_name}_Metrics': metric_fig,
        # }

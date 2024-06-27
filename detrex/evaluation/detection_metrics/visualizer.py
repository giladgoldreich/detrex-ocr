import os
import logging
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc as area_under_curve
from detectron2.utils.visualizer import Visualizer as D2Visualizer


class Visualizer(D2Visualizer):

    @staticmethod
    def precision_recall_curve(precision, recall, legend_list=None):
        """
        Receives two vectors, recall and precision
        Plots the PR curve for them
        :param np.ndarray precision: A 1D or 2D array with the precision values in its columns
        :param np.ndarray recall: A 1D or 2D array with the recall values in its columns
        :param legend_list: Either a list
        :return: The precision-recall figure
        """
        # We don't want to break training because of a bad figure
        sns.set(style="darkgrid")
        pr_fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        try:
            precision_array = np.array(precision).squeeze()
            recall_array = np.array(recall).squeeze()
            assert precision_array.shape == recall_array.shape

            # Sorting according to the recall
            precision_list, recall_list, auc_list = list(), list(), list()
            if len(recall_array.shape) == 1:
                precision_list = [precision_array]
                recall_list = [recall_array]
            else:
                precision_list = [precision_array[:, i] for i in range(recall_array.shape[1])]
                recall_list = [recall_array[:, i] for i in range(recall_array.shape[1])]

            for recall, precision in zip(precision_list, recall_list):
                # We have to sort the x-axis so that the auc computation will be defined
                recall_sort_map = recall.argsort()
                recall = recall[recall_sort_map]
                precision = precision[recall_sort_map]
                auc_list.append(area_under_curve(x=recall, y=precision))
                sns.lineplot(x=recall, y=precision, ax=ax)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            if legend_list is not None:
                plt.legend([f'{name}: {auc:.4f}' for name, auc in zip(legend_list, auc_list)])
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f'Failed to produce PR curve. Exception: {e}')
        return pr_fig

    @staticmethod
    def metric_summary_plot(metrics_df):
        """
        Plots precision recall and fscore over the entire score range
        :param pd.DataFrame metrics_df: The summary dataframe produced by OCRMetrics
        :return:
        """
        sns.set(style="darkgrid")
        columns = ['fscore', 'recall', 'precision']
        metric_fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        for y in columns:
            sns.lineplot(data=metrics_df, x='score', y=y, ax=axes[0])
        axes[0].legend(columns)

        for y in columns:
            sns.lineplot(data=metrics_df, x='score', y=y + '_err', ax=axes[1])
        axes[1].set(yscale="log")
        axes[1].legend(columns)
        for ax in axes:
            ax.set_ylabel('')
        return metric_fig

    def plot_preds_and_gt(self, image_path, pred_boxes, gt_boxes, output_dir):
        image = np.array(Image.open(image_path))
        v_gt = self.__class__(image, None)
        v_gt = v_gt.overlay_instances(boxes=gt_boxes.tensor.cpu().numpy())
        anno_img = v_gt.get_image()
        max_vis_boxes = min(len(pred_boxes), 200)
        v_pred = self.__class__(image, None)
        v_pred = v_pred.overlay_instances(
            boxes=pred_boxes[0:max_vis_boxes].tensor.cpu().numpy()
        )
        prop_img = v_pred.get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_vis.jpg')
        Image.fromarray(vis_img).save(vis_path)

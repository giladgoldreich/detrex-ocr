import itertools
import logging
import os
import pickle
from datetime import datetime
from copy import deepcopy
import pandas as pd
import torch
from tabulate import tabulate
from termcolor import colored
from torch.utils.tensorboard._utils import figure_to_image

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.events import get_event_storage
# from .attribute_metrics import AttributeMetrics
# from .bezier_metrics import BezierMetrics
from .box_metrics import BoxMetrics
from .eval_vis_handler import EvalVisHandler

# from .exact_line_metrics import ExactLineMetrics
# from .filtered_box_metrics import FilteredBoxMetrics
# from .recognition_metrics import RecognitionMetrics
# from .filtered_recognition_metrics import FilteredRecognitionMetrics
# from .academic_recognition_metrics import AcademicRecognitionMetrics
# from ..data.text_encoder import TextEncoder
# from ..postprocess.post_processor import build_post_processor


class OCREvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, 
                 cfg, 
                 distributed=True, 
                 output_dir=None, 
                 iou_threshold=None,
                 vis_flag=True, 
                 save_flag=False, 
                 verbose=True, 
                 is_train=True,
                 rotated_boxes=False,
                 iou_types=('iou'),
                 plot_eval_on=True,
                 plot_assets_delta=10,
                 plot_assets_max_images=100,
                 class_names = ('text',)
                 ):
        """
        Args:
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self.logger = logging.getLogger(__name__)
        if iou_threshold is not None:
            self.logger.warning('IOU Threshold for ocr_evaluator was initialized using the config')

        self.cfg = cfg
        self.vis_flag = vis_flag
        self.verbose = verbose
        self.save_flag = save_flag
        self.train = is_train
        self._distributed = distributed
        self._output_dir = output_dir
        
        # gilad - taking everything from input and not from config
        
        self.rotated_boxes = cfg.MODEL.ROTATED_BOXES_ON if cfg is not None else rotated_boxes
        self.iou_types = self.cfg.TEST.IOU_TYPES if cfg is not None else iou_types  # ['iou', 'tight_iou', 'rotated_iou']
        # plot res on images params
        self.plot_eval_on_images = self.cfg.TEST.PLOT_EVAL_ON if cfg is not None else plot_eval_on
        self.plot_assets_delta = self.cfg.TEST.PLOT_ASSETS_DELTA if cfg is not None else plot_assets_delta
        self.plot_assets_max_images = self.cfg.TEST.PLOT_ASSETS_MAX_IMAGES if cfg is not None else plot_assets_max_images
        self.plot_assets_list = list()

        self.asset_counter = 0

        self._cpu_device = torch.device("cpu")

        self.class_names = self.cfg.MODEL.ROI_HEADS.CLASS_NAMES if cfg is not None else list(class_names)
        
        # gilad -post processing - disabled from copy
        self.post_processor = lambda x: x
        # # Todo (amirmak): flag in Test to skip PP if needed
        # if cfg.POST_PROCESSING.SKIP_ALL:
        #     self.post_processor = lambda x: x
        # else:
        #     self.post_processor = build_post_processor(cfg, build_single_word_lines=False, verbose=False)

        # Adding the box metrics for the individual classes
        self._class_box_metrics = list()
        # Adding a list for metrics that happen after post-processing
        self._class_box_pp_metrics = list()
        # other metrics
        self._other_metrics = list()

        # gilad - disable text encode and such
        # if cfg.TEST.USE_FILTERED_METRICS:
        #     decode_func = TextEncoder(cfg).decode

        for i, class_name in enumerate(self.class_names):
            for iou_type in self.iou_types:
                name = class_name.title() + '_' + iou_type.title()
                self._class_box_metrics.append(BoxMetrics(cfg=cfg, iou_type=iou_type, box_class=i, name=name))
                
                # gilad - no filtered metrics

                # if cfg.TEST.USE_FILTERED_METRICS:
                #     self._class_box_metrics.append(FilteredBoxMetrics(cfg=cfg, iou_type=iou_type, box_class=i,
                #                                                       name=name + '_Filtered', decode_func=decode_func))

                # if (not self.train) and (not cfg.POST_PROCESSING.SKIP_ALL):
                #     name = class_name.title() + '_' + iou_type.title() + '_PostProcess'
                #     self._class_box_pp_metrics.append(BoxMetrics(cfg=cfg, iou_type=iou_type, name=name,
                #                                                  default_score=0,
                #                                                  len_scores_vec=1,
                #                                                  box_class=i))

        # gilad - skipping all other metrics
        # if not cfg.POST_PROCESSING.SKIP_ALL and not self.train:
        #     if 'line' in self.class_names:
        #         line_class = self.class_names.index('line')
        #         self._other_metrics.append(ExactLineMetrics(cfg=cfg, post_processor=self.post_processor,
        #                                                     line_class=line_class, name='Line_Exact_Match'))

        # #  Adding attribute metrics
        # if cfg.MODEL.ATTRIBUTE_ON:
        #     self._other_metrics.append(AttributeMetrics(cfg=cfg, name='Attributes'))

        # # Adding Bezier metrics
        # if getattr(cfg.MODEL, 'BORDER', None):
        #     border_bezier = cfg.MODEL.BORDER.BEZIER_ON
        # else:
        #     border_bezier = False
        # if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "BezierFCOS" or \
        #         cfg.MODEL.PROPOSAL_GENERATOR.NAME == "BezierTextHead" or \
        #         cfg.MODEL.BEZIER.BEZIER_ON or \
        #         border_bezier:
        #     if not self.train:
        #         self._other_metrics.append(BezierMetrics(cfg=cfg, bezier_class=0, name='Word_Bezier_IOU'))
        #         self._other_metrics.append(BezierMetrics(cfg=cfg, bezier_class=1, name='Line_Bezier_IOU'))

        # if cfg.MODEL.RECOGNIZER_ON:
        #     box_class = self.class_names.index('word')
        #     iou_type = 'iou'
        #     name = 'Word_' + iou_type.title() + '_E2E'
        #     self._other_metrics.append(RecognitionMetrics(cfg=cfg,
        #                                                   iou_type='iou',
        #                                                   box_class=box_class,
        #                                                   name=name
        #                                                   ))
        #     if cfg.TEST.USE_FILTERED_METRICS:
        #         box_class = self.class_names.index('word')
        #         iou_type = 'iou'
        #         name = 'Word_' + iou_type.title() + '_E2E' + '_Filtered'
        #         self._class_box_metrics.append(FilteredRecognitionMetrics(cfg=cfg,
        #                                                   iou_type='iou',
        #                                                   box_class=box_class,
        #                                                   word_spotting=False,
        #                                                   name=name
        #                                                   ))
        #         name = 'Word_' + iou_type.title() + '_E2E' + '_Academic'
        #         self._class_box_metrics.append(AcademicRecognitionMetrics(cfg=cfg,
        #                                                   iou_type='iou',
        #                                                   box_class=box_class,
        #                                                   word_spotting=False,
        #                                                   name=name
        #                                                   ))
        #         name = 'Word_' + iou_type.title() + '_WordSpotting' + '_Academic'
        #         self._class_box_metrics.append(AcademicRecognitionMetrics(cfg=cfg,
        #                                                   iou_type='iou',
        #                                                   box_class=box_class,
        #                                                   word_spotting=True,
        #                                                   name=name
        #                                                   ))

        if (save_flag or self.plot_eval_on_images or ~self.train) and (
                not os.path.isdir(self._output_dir)) and comm.is_main_process():
            os.makedirs(self._output_dir, exist_ok=True)
            
        if comm.is_main_process():
            os.makedirs(os.path.join(self._output_dir, 'eval_metrics'), exist_ok=True)

    def reset(self):
        for metric in self._class_box_metrics:
            metric.reset()
        for metric in self._class_box_pp_metrics:
            metric.reset()
        for metric in self._other_metrics:
            metric.reset()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred_instances = output['instances']
            dataset_name = input['dataset_name']
            file_name = input['file_name']
            gt_instances = input['instances']
            gt_instances = gt_instances.to(pred_instances.pred_boxes.tensor.get_device())

            # Scaling the GT instances to match the size of predictions
            gt_scale = pred_instances.image_size[0] / gt_instances.image_size[0]
            gt_instances.gt_boxes.scale(gt_scale, gt_scale)
            if gt_instances.has('gt_masks'):
                for mask in gt_instances.gt_masks.polygons:
                    mask[0] = mask[0] * gt_scale

            self.asset_counter += 1
            # Adding the document to our metrics container
            for metric in self._class_box_metrics:
                metric.add_entry(dataset_name, file_name, pred_instances=pred_instances, gt_instances=gt_instances)

            if len(self._other_metrics) > 0:
                for metric in self._other_metrics:
                    metric.add_entry(dataset_name, file_name, pred_instances=pred_instances, gt_instances=gt_instances)

            if len(self._class_box_pp_metrics) > 0:
                pp_pred_instances = self.post_processor(pred_instances)
                for pp_metric in self._class_box_pp_metrics:
                    pp_metric.add_entry(dataset_name, file_name, pred_instances=pp_pred_instances,
                                        gt_instances=gt_instances)

            if self.plot_eval_on_images and (self.asset_counter % self.plot_assets_delta == 0) and (
                    len(self.plot_assets_list) < self.plot_assets_max_images):
                asset_id = os.path.splitext(os.path.basename(file_name))[0]
                self.plot_assets_list.append(dict(
                    file_path=file_name,
                    asset_id=asset_id,
                    pred_instances=pred_instances.to('cpu'),
                    gt_instances=gt_instances.to('cpu')))

    def evaluate(self):
        if self._distributed:
            self.logger.info('Synchronizing and gathering predictions')
            comm.synchronize()
            metrics = comm.gather(self._class_box_metrics, dst=0)
            pp_metrics = comm.gather(self._class_box_pp_metrics, dst=0)
            for met in self._other_metrics:
                # remove post processors to allow operations on metrics
                met.post_processor = None
            other_metrics = comm.gather(self._other_metrics, dst=0)
            if self.plot_eval_on_images:
                plot_assets_list = comm.gather(self.plot_assets_list, dst=0)

            if not comm.is_main_process():
                return {}
            # Unzipping and summing the metrics
            self._class_box_metrics = [sum(metrics) for metrics in list(zip(*metrics))]
            self._class_box_pp_metrics = [sum(metrics) for metrics in list(zip(*pp_metrics))]
            self._other_metrics = [sum(metrics) for metrics in list(zip(*other_metrics))]
            if self.plot_eval_on_images:
                self.plot_assets_list = list(itertools.chain(*plot_assets_list))

        # From now on this code runs only on the main process
        all_classes_iou = 'iou' if 'iou' in self.iou_types else self.iou_types[0]
        all_classes_boxes_metric = self._combine_class_metrics(metrics=self._class_box_metrics,
                                                               iou=all_classes_iou,
                                                               metric_name='All_Classes')

        all_metrics_dict = dict()
        table_dict = dict()
        for metric in [*self._class_box_metrics, *self._class_box_pp_metrics, *self._other_metrics,
                       *all_classes_boxes_metric]:
            metrics_dict = metric.summary_dict()
            if self.verbose:
                summary_table = tabulate(metric.summary_table(), headers='firstrow', floatfmt='.4f')
                self.logger.info(f'Eval Results for {colored(metric.metric_name, attrs=["bold", "underline"])}:\n'
                                 f'{colored(summary_table, "cyan")}')
            if self.vis_flag:
                try:
                    storage = get_event_storage()
                    for figure_name, figure in metric.produce_figure_dict().items():
                        storage.put_image(figure_name, figure_to_image(figure))
                    pr_fig = EvalVisHandler.plot_combined_precision_recall_curve(metrics=self._class_box_metrics)
                    storage.put_image('PR_Curve', figure_to_image(pr_fig))
                except AssertionError as e:
                    self.logger.warning(f'Could not visualize {metric.metric_name}: {e}')

            if self.save_flag:
                try:
                    iteration = f'_{get_event_storage().iter}'
                except AssertionError:
                    iteration = ''
                with open(
                        os.path.join(self._output_dir, 'eval_metrics', f'metrics{iteration}_{metric.metric_name}.pkl'),
                        'wb') as fp:
                    pickle.dump(metric, fp)

            table_dict.update({f'{metric.metric_name}': deepcopy(metrics_dict)})
            # We provide a little hierarchy for easy navigation of the metrics
            if 'ALL' in metrics_dict:
                all_metrics_dict.update({f'val_{metric.metric_name}': metrics_dict.pop('ALL'),
                                         f'val_{metric.metric_name}_Datasets': metrics_dict})
            else:
                all_metrics_dict.update({f'{metric.metric_name}': metrics_dict})
                

        if self.plot_eval_on_images:
            EvalVisHandler.visualize_assets(asset_list=self.plot_assets_list,
                                            class_names=self.class_names,
                                            output_dir=self._output_dir)

        if not self.train:
            dataset_dict = []
            for met, d in table_dict.items():
                for key, val in d.items():
                    val.update({'metric': met, 'dataset': key})
                    dataset_dict.append(val)
            df = pd.DataFrame(dataset_dict)
            date_string = datetime.now().strftime("%Y%m%d_%H%M")
            df.to_excel(os.path.join(self._output_dir, 'eval_metrics', "metrics_{}.xlsx".format(date_string)))
        
        return all_metrics_dict

    @staticmethod
    def _combine_class_metrics(metrics, iou, metric_name):
        filt_metrics = [x for x in metrics if x.iou_type == iou]
        if len(filt_metrics) < 1:
            return list()
        new_comb_metric = filt_metrics[0]
        for tmp_metric in filt_metrics[1:]:
            new_comb_metric = new_comb_metric.__add__(tmp_metric)

        new_comb_metric.metric_name = metric_name
        return [new_comb_metric]

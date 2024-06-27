from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, DatasetEvaluator
from detrex.evaluation.vis_evaluator import VisulizationEvaluator
from typing import Union, List
from pathlib import Path
import os
from .detection_metrics.ocr_evaluator import OCREvaluator


def create_evaluator(dataset_name: Union[str, List[str]],
                     output_dir: Union[str, Path] = './eval',
                     vis: bool = True,
                     coco: bool = False,
                     box: bool = True,
                     max_dets_per_image: int = 2000,
                     score_threshold: float = 0.4,
                     iou_threshold=0.5,
                     ) -> DatasetEvaluator:
    evaluator_list = []
    if coco:
        evaluator_list.append(
            COCOEvaluator(dataset_name=dataset_name, max_dets_per_image=max_dets_per_image,
                          output_dir=os.path.join(output_dir, 'coco'))
        )

    if vis:
        evaluator_list.append(
            VisulizationEvaluator(output_dir=output_dir,
                                  score_threshold=score_threshold,
                                  remove_classes=True,
                                  remove_scores=True,
                                  only_boxes=True,
                                  font_size=1)
        )
        
    if box:
        evaluator_list.append(OCREvaluator(
            cfg=None,
            output_dir=output_dir,
            iou_threshold=iou_threshold,
            iou_types=['iou'],
            is_train=False,
            rotated_boxes=False,
            plot_eval_on=False,
            save_flag=True
        ))

    return DatasetEvaluators(evaluator_list)

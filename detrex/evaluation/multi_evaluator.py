from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, DatasetEvaluator
from detrex.evaluation.vis_evaluator import VisulizationEvaluator
from typing import Union
from pathlib import Path
import os


def create_evaluator(dataset_name: str,
                     output_dir: Union[str, Path] = './eval',
                     vis: bool = True,
                     coco: bool = False,
                     max_dets_per_image: int = 2000,

                     ) -> DatasetEvaluator:
    evaluator_list = []
    if coco:
        evaluator_list.append(
            COCOEvaluator(dataset_name=dataset_name, max_dets_per_image=max_dets_per_image,
                          output_dir=os.path.join(output_dir, dataset_name, 'coco'))
        )

    if vis:
        evaluator_list.append(
            VisulizationEvaluator(output_dir=os.path.join(output_dir, dataset_name),
                                  score_threshold=0.5,
                                  remove_classes=True,
                                  remove_scores=True,
                                  only_boxes=True,
                                  font_size=1)
        )

    return DatasetEvaluators(evaluator_list)

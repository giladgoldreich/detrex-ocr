from detectron2.evaluation import DatasetEvaluator
import logging
from pathlib import Path
from typing import Union
from detectron2.utils.file_io import PathManager
from detectron2.utils import comm
import torch
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask, ColorMode
import numpy as np
import os
import cv2
from torchvision.transforms import functional as F
import warnings
from copy import deepcopy

class VisulizationEvaluator(DatasetEvaluator):
    
    def __init__(self,
                 output_dir: Union[str, Path],
                 add_vis_suffix: bool = True,
                 score_threshold: float = 0.5,
                 remove_classes: bool = True,
                 remove_scores: bool = True,
                 font_size: int = 1,
                 only_boxes: bool = True):
        
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._output_dir = Path(output_dir)
        if add_vis_suffix:
            self._output_dir = self._output_dir / 'vis'
        self._score_threshold = score_threshold
        self._remove_classes = remove_classes
        self._remove_scores: bool = remove_scores
        self._font_size = font_size
        self._only_boxes = only_boxes
        
        if not self._only_boxes:
            warnings.warn("It is highly recommended to use the `only_boxes` mode as the `Visualizer` class is super slow")
        
        if comm.is_main_process():
            PathManager.mkdirs(str(self._output_dir))
        comm.synchronize()
        self._cpu_device = torch.device('cpu')
        
    
    def process(self, inputs, outputs):
        for input_dict, output_dict in zip(inputs, outputs):
            image = input_dict['image']
            image = F.to_pil_image(image).convert('RGB')
            image = np.array(image)
            origin_image = image.copy()
            vis_image = origin_image.copy()
            
                      
            input_instances = deepcopy(input_dict['instances'].to(self._cpu_device))
            vis_image = self.visualize_input_instances(input_instances, vis_image)
            
            if 'instances' in output_dict:
                # the instances in the output dictionary are in the original images coordinates. Therefore, one needs to normalize them and scale them.
                detected_instances = deepcopy(output_dict['instances'].to(self._cpu_device))
                scale = input_instances.image_size[0] / detected_instances.image_size[0]
                detected_instances.pred_boxes.scale(scale, scale)
                if detected_instances.has('pred_masks'):
                    for mask in detected_instances.pred_masks:
                        mask[0] = mask[0] * scale
                if detected_instances.has('pred_keypoints'):
                    raise RuntimeError('Cannot scale predicted keypoints')
                
                filtered_detected_instances = detected_instances[detected_instances.scores >= self._score_threshold]
                vis_image = self.visualize_det_instances(filtered_detected_instances, vis_image)
                                                
            # canvas = np.hstack([origin_image, vis_image])
            canvas = vis_image           
                
            basename = os.path.basename(input_dict["file_name"])
            dataset_name = input_dict.get('dataset_name', None)
            outdir = Path(self._output_dir)
            if dataset_name is not None:
                outdir = outdir / dataset_name
                PathManager.mkdirs(str(outdir))
                
            outpath = Path(outdir / basename).with_suffix('.jpg')
            cv2.imwrite(str(outpath), canvas[..., ::-1])
        
        
    def reset(self):
        pass
    
    def evaluate(self):
        return {}
    
    def visualize_det_instances(self, predictions: Instances, img: np.ndarray) -> np.ndarray:
        
        if self._only_boxes:
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            if boxes is None or len(boxes) == 0:
                return img

            for b in boxes.tensor.detach().numpy().astype(int):
                cv2.rectangle(img, b[:2], b[2:], color=(0, 255, 0), thickness=self._font_size)
            return img
                
                
        vis = Visualizer(img)
        vis._default_font_size = self._font_size
        
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes if not self._remove_classes else None, 
                                     scores if not self._remove_scores else None,
                                     vis.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, vis.output.height, vis.output.width) for x in masks]
        else:
            masks = None

        if vis._instance_mode == ColorMode.IMAGE_BW:
            vis.output.reset_image(
                vis._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3
        
        else:
            alpha = 0.5
        
        colors = [[0., 1., 0.] for b in boxes]        

        vis.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return vis.output.get_image()
    
    
    def visualize_input_instances(self, input_instances: Instances, img: np.ndarray) -> np.ndarray:
        
        if self._only_boxes:
            boxes = input_instances.gt_boxes if input_instances.has("gt_boxes") else None
            if boxes is None or len(boxes) == 0:
                return img

            for b in boxes.tensor.detach().numpy().astype(int):
                cv2.rectangle(img, b[:2], b[2:], color=(255, 0, 0), thickness=self._font_size)
            return img
        
        vis = Visualizer(img)
        vis._default_font_size = self._font_size

                
        boxes = input_instances.gt_boxes if input_instances.has("gt_boxes") else None        
        classes = input_instances.gt_classes.tolist() if input_instances.has("gt_classes") else None
        labels = _create_text_labels(classes if not self._remove_classes else None, 
                                     None,
                                     vis.metadata.get("thing_classes", None))
        keypoints = input_instances.gt_keypoints if input_instances.has("gt_keypoints") else None

        if input_instances.has("gt_masks"):
            masks = np.asarray(input_instances.gt_masks)
            masks = [GenericMask(x, vis.output.height, vis.output.width) for x in masks]
        else:
            masks = None

        if vis._instance_mode == ColorMode.IMAGE_BW:
            vis.output.reset_image(
                vis._create_grayscale_image(
                    (input_instances.gt_masks.any(dim=0) > 0).numpy()
                    if input_instances.has("gt_masks")
                    else None
                )
            )
            alpha = 0.3
        
        else:
            alpha = 0.5
        
        colors = [[1., 0., 0.] for b in boxes]        

        vis.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return vis.output.get_image()
        
                    

        
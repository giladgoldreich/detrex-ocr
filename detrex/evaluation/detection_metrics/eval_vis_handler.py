import numpy as np
import os
from .box_metrics import BoxMetrics
from .visualizer import Visualizer
from detectron2.utils.visualizer import Visualizer as D2Visualizer
from detectron2.utils.visualizer import _create_text_labels
from detectron2.utils.events import get_event_storage
from PIL import Image
import matplotlib


class EvalVisHandler:
    
    @staticmethod
    def visualize_assets(asset_list, class_names, output_dir=None):
        for asset_dict in asset_list:
            asset_id = asset_dict['asset_id']
            EvalVisHandler.visualize_single_asset(asset_dict=asset_dict,
                                                  class_names=class_names,
                                                  title=f'Val_Image_{asset_id}',
                                                  output_dir=output_dir)

    @staticmethod
    def visualize_single_asset(asset_dict, class_names=None, title='', min_pred_score=0.35, output_dir=None,
                               max_size=1200):
        image_path = asset_dict['file_path']
        gt_instances = asset_dict['gt_instances']
        pred_instances = asset_dict['pred_instances']
        pred_instances = pred_instances[pred_instances.scores >= min_pred_score]
        vis_img = EvalVisHandler.visualize_preds_and_gt(image_path=image_path,
                                                        pred_instances=pred_instances,
                                                        gt_instances=gt_instances,
                                                        class_names=class_names)

        pil_image = Image.fromarray(vis_img.transpose((1, 2, 0)))
        if output_dir is not None:
            try:
                iteration = get_event_storage().iter
                img_output_dir = os.path.join(output_dir, 'eval_images', f'{title}')
                if not os.path.isdir(img_output_dir):
                    os.makedirs(img_output_dir)
                output_path = os.path.join(img_output_dir, f'{title}_itr_{iteration:09d}.jpg')
                if os.path.isfile(output_path):
                    title = title + '-b'
                    output_path = os.path.join(img_output_dir, f'{title}_itr_{iteration:09d}.jpg')
                pil_image.save(output_path)
            except AssertionError:
                pass

        pil_image.thumbnail((max_size, max_size))
        vis_img = np.array(pil_image).transpose((2, 0, 1))
        storage = get_event_storage()
        storage.put_image(title, vis_img)

    @staticmethod
    def visualize_preds_and_gt(image_path, pred_instances, gt_instances, class_names=None):
        if len(class_names) > 2:
            cmap = matplotlib.cm.get_cmap('gist_ncar')
        else:
            cmap = matplotlib.cm.get_cmap('winter')
        class_colors = cmap(np.linspace(0, 1, len(class_names)))

        img = np.array(Image.open(image_path).convert('RGB'))
        v_gt = D2Visualizer(img, None)
        v_pred = D2Visualizer(img, None)

        # visualize gt
        if len(class_names) > 2:
            labels = [class_names[x] for x in gt_instances.gt_classes]
        else:
            labels = None

        if (gt_instances is not None) and len(gt_instances) > 0:
            gt_colors = [class_colors[x] for x in gt_instances.gt_classes]
            v_gt = v_gt.overlay_instances(boxes=gt_instances.gt_boxes.tensor.cpu().numpy(),
                                          labels=labels,
                                          assigned_colors=gt_colors,
                                          alpha=0.5)

        try:
            anno_img = v_gt.get_image()
        except:
            anno_img = img

        # visualize pred
        if (pred_instances is not None) and len(pred_instances) > 0:

            # TODO: tsiper: The labels are too noisy for visualization
            labels = None
            # scores = pred_instances.scores if pred_instances.has("scores") else None
            # classes = pred_instances.pred_classes if pred_instances.has("pred_classes") else None
            #
            # if len(class_names) > 2:
            #     labels = _create_text_labels(classes, scores, class_names)
            # else:
            #     labels = _create_text_labels(classes, scores, None)

            pred_colors = [class_colors[x] for x in pred_instances.pred_classes] if pred_instances.has(
                "pred_classes") else None
            v_pred = v_pred.overlay_instances(
                boxes=pred_instances.pred_boxes.tensor.cpu().numpy(),
                labels=labels,
                assigned_colors=pred_colors,
                alpha=0.5)
        try:
            prop_img = v_pred.get_image()
        except:
            prop_img = img

        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose((2, 0, 1))
        return vis_img

    @staticmethod
    def plot_combined_precision_recall_curve(metrics):
        recall_list, precision_list, legend_list = list(), list(), list()
        box_metrics = [metric for metric in metrics if isinstance(metric, BoxMetrics)]
        for metric in box_metrics:

            if metric.recall_vec is not None and metric.precision_vec is not None:
                recall_list.append(np.array(metric.recall_vec).squeeze())
                precision_list.append(np.array(metric.precision_vec).squeeze())
                legend_list.append(metric.metric_name)
        if recall_list and precision_list:
            recall = np.stack(recall_list).squeeze().T
            precision = np.stack(precision_list).squeeze().T
            pr_fig = Visualizer.precision_recall_curve(precision=precision, recall=recall, legend_list=legend_list)
            return pr_fig

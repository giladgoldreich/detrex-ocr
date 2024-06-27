from detrex.config import get_config
from ..models.dino_r50 import model
import copy
import os

# get default config
dataloader = get_config("common/data/pretrain.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
dataloader.train.dataset.weights = [1.] # uniform weighting
# train.output_dir = "./output/train_bs16_clean_equal_weight"
train.output_dir = "./output/test_gilad"


# todo - check this checkpointing module, may save memory?
# model.backbone.use_checkpoint=True
model.transformer.encoder.use_checkpoint=True
model.transformer.decoder.use_checkpoint=True

# max training iterations
train.max_iter = 90000 # ~100 epochs on clean pretrain w/o textocr when bs=16 (|clean_ds|~14500 samples w/o textocr)
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000
train.seed = 42

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device
model.num_classes = 1
model.num_queries = 3000
model.dn_number = 300
model.select_box_nums_for_evaluation = 2000
model.vis_period = 0 # takes too much memory in tensorboard


# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.max_dets_per_image = model.select_box_nums_for_evaluation
dataloader.evaluator.output_dir = os.path.join(train.output_dir, 'eval')

# better hyperparms from detrex repo
# no frozen backbone get better results
model.backbone.freeze_at = -1

# use 2.0 for class weight
model.criterion.weight_dict = {
    "loss_class": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_class_dn": 1,
    "loss_bbox_dn": 5.0,
    "loss_giou_dn": 2.0,
}

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

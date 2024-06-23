from detrex.config import get_config
from ..models.dino_r50 import model

# get default config
dataloader = get_config("common/data/pretrain.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/ddi_100_dino_r50_4scale_30000_steps_noeval"

# max training iterations
train.max_iter = 30000
train.eval_period = 30000
train.log_period = 20
train.checkpointer.period = 2500
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
model.vis_period = 20


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
dataloader.evaluator.output_dir = train.output_dir
dataloader.evaluator.max_dets_per_image = model.select_box_nums_for_evaluation

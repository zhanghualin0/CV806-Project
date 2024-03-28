import torch
from pytorch_lightning import LightningModule

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()

model = MyModel.load_from_checkpoint('./outputs/cirr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-0.0001/base/ckpt_5.ckpt')

model_state_dict = model.state_dict()

torch.save(model_state_dict, './outputs/cirr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-0.0001/base/ckpt_cirr.pth')

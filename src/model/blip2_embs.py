import torch
from torch import nn
import logging

from src.model.blip2 import load_checkpoint, init_Qformer, init_tokenizer, create_eva_vit_g
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)

# from src.model.blip import create_vit, init_tokenizer, load_checkpoint
# from src.model.med import BertConfig, BertModel


class BLIP2Embs(nn.Module):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=384,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.visual_encoder, self.ln_vision = create_eva_vit_g(
            vit_model,img_size,drop_path_rate,use_grad_checkpoint,vit_precision
        )

        self.freeze_vit = freeze_vit
        if self.freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )

        self.tokenizer = init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # state_dict = self.Qformer.state_dict()
        # for name, param in self.Qformer.named_parameters():
        #     if "_query" in name:
        #         key_orig = name.replace("_query", "")
        #         param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # for p in self.vision_proj.parameters():
        #     p.requires_grad = False

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len


def blip2_embs(pretrained="", **kwargs):
    model = BLIP2Embs(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
        # assert len(msg.missing_keys) == 0, "Missing keys!"
    return model

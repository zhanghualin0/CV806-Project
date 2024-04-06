from typing import Any
import logging

import einops
import torch
import torch.nn.functional as F
from torch import nn

# from src.model.blip2_qformer import Blip2Qformer 
from src.model.blip2 import load_checkpoint, init_Qformer, init_tokenizer, create_eva_vit_g
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from src.tools.utils import print_dist

class BLIP2Cir(nn.Module):
    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",
        img_size=224,
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

        self.loss = loss

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
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, batch, fabric=None):
        ref_image, target_features, caption, _ = batch

        device = ref_image.device

        # if self.freeze_vit:
        #     with torch.no_grad():
        #         image_embeds_frozen = self.ln_vision(self.visual_encoder(ref_image))
        # else:
        #     image_embeds_frozen = self.ln_vision(self.visual_encoder(ref_image))

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            if self.freeze_vit:
                with torch.no_grad():
                    image_embeds_frozen = self.ln_vision(self.visual_encoder(ref_image))
            else:
                image_embeds_frozen = self.ln_vision(self.visual_encoder(ref_image))
            
        image_embeds_frozen = image_embeds_frozen.float()
        # print("imae_embed size:", image_embeds_frozen.shape) --> torch.Size([256, 730, 1408])
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

        text = self.tokenizer(caption, return_tensors="pt", padding=True).to(device)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

        output = self.Qformer.bert(
            text.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        multimodal_embeds = output.last_hidden_state[:, 0, :]  
        # print("ssssssssizeeeeee", multimodal_embeds.shape)   --> 768

        # print("hidden size", self.Qformer.config.hidden_size) --> 768

        multimodal_embeds = F.normalize(self.text_proj(multimodal_embeds), dim=-1)     
        # print("ssssssssizeeeeee", multimodal_embeds.shape) 

        target_features = target_features.to(device)
        target_features = F.normalize(target_features, dim=-1)

        if fabric.world_size > 1:
            multimodal_embeds = fabric.all_gather(multimodal_embeds, sync_grads=True)
            multimodal_embeds = einops.rearrange(multimodal_embeds, "d b e -> (d b) e")

            target_features = fabric.all_gather(target_features, sync_grads=True)
            target_features = einops.rearrange(target_features, "d b e -> (d b) e")

        # Calculate loss between multimodal features and target features
        return self.loss(multimodal_embeds, target_features, self.temp)

def blip2_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        print("loading checkpoint >>>")
        model, msg = load_checkpoint(model, ckpt_path)
        # for name, param in model.named_parameters():
        #     print("keys:", name)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model

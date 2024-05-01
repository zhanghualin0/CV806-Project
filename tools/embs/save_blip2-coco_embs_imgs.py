import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from src.data.embs import ImageDataset
from src.model.blip2_embs import blip2_embs # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main(args):
    dataset = ImageDataset(
        image_dir=args.image_dir,
        img_ext=args.img_ext,
        save_dir=args.save_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")
    model = blip2_embs(
        pretrained="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_finetune_coco.pth",
        vit_model="eva_clip_g",
        img_size=364,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    )

    model = model.to(device)
    model.eval()

    for imgs, video_ids in tqdm(loader):
        imgs = imgs.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            img_embs = model.ln_vision(model.visual_encoder(imgs))
        
        img_atts = torch.ones(img_embs.size()[:-1], dtype=torch.long).to(imgs.device)

        query_tokens = model.query_tokens.expand(img_embs.shape[0], -1, -1)

        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=img_embs,
            encoder_attention_mask=img_atts,
            use_cache=True,
            return_dict=True,
        )

        img_feats = F.normalize(model.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1).cpu()
        # print(img_feats.shape)
        # print(type(img_feats))

        for img_feat, video_id in zip(img_feats, video_ids):
            torch.save(img_feat, args.save_dir / f"{video_id}.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Path to image directory"
    )
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--img_ext", type=str, default="png")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="coco"
    )
    args = parser.parse_args()

    subdirectories = [subdir for subdir in args.image_dir.iterdir() if subdir.is_dir()]
    if len(subdirectories) == 0:
        args.save_dir = args.image_dir.parent / f"blip2-embs-{args.model_type}"
        args.save_dir.mkdir(exist_ok=True)
        main(args)
    else:
        for subdir in subdirectories:
            args.image_dir = subdir
            args.save_dir = (
                subdir.parent.parent / f"blip2-embs-{args.model_type}" / subdir.name
            )
            args.save_dir.mkdir(exist_ok=True, parents=True)

            if subdir.name == 'train':
                for subsubdir in subdir.iterdir():
                    args.image_dir = subsubdir
                    args.save_dir = (
                        subdir.parent.parent / f"blip2-embs-{args.model_type}" / subdir.name / subsubdir.name
                    )
                    args.save_dir.mkdir(exist_ok=True, parents=True)    
                    main(args)
            else:
                main(args)

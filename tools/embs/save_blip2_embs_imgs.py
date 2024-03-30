import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# Ensure lavis is installed or accessible in your environment
from lavis.models import load_model_and_preprocess
from torchvision.transforms.functional import to_pil_image

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from src.data.embs import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Load BLIP2 model and preprocessors
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    model.eval()

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

    print("Processing images for embeddings")
    for images, image_ids in tqdm(loader):
        # Process and batch images
        images = torch.stack([
            vis_processors["eval"](to_pil_image(img)).unsqueeze(0) for img in images
            ]).to(device)

        sample = {"image": images}
        sample["image"] = sample["image"].squeeze(1)
        features_image = model.extract_features(sample, mode="image")
        features_image = F.normalize(model.vision_proj(features_image.image_embeds[:, 0, :]), dim=-1).cpu()

        # print(features_image.image_embeds.shape)
        # print(type(features_image))

        for img_feat, image_id in zip(features_image, image_ids):
            # img_feat = img_feat.squeeze(0)  # Remove batch dimension if any
            # img_feat = img_feat.to(torch.float16)
            torch.save(img_feat, os.path.join(args.save_dir, f"{image_id}.pth"))

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

    args = parser.parse_args()

    subdirectories = [subdir for subdir in args.image_dir.iterdir() if subdir.is_dir()]
    if len(subdirectories) == 0:
        args.save_dir = args.image_dir.parent / f"blip2-embs"
        args.save_dir.mkdir(exist_ok=True)
        main(args)
    else:
        for subdir in subdirectories:
            args.image_dir = subdir
            args.save_dir = (
                subdir.parent.parent / f"blip2-embs" / subdir.name
            )
            args.save_dir.mkdir(exist_ok=True, parents=True)

            if subdir.name == 'train':
                for subsubdir in subdir.iterdir():
                    args.image_dir = subsubdir
                    args.save_dir = (
                        subdir.parent.parent / f"blip2-embs" / subdir.name / subsubdir.name
                    )
                    args.save_dir.mkdir(exist_ok=True, parents=True)    
                    main(args)
            else:
                main(args)

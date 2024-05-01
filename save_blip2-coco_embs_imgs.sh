#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4



# Run the Python script with the necessary arguments
python ./tools/embs/save_blip2-coco_embs_imgs.py \
    --image_dir ./datasets/CIRR/images \
    --img_ext png \
    --batch_size 128 \
    --num_workers 4 


# #!/bin/bash



# Run the Python script with the necessary arguments
python ./tools/embs/save_blip2-clipL_embs_imgs.py \
    --image_dir ./datasets/CIRR/images \
    --img_ext png \
    --batch_size 128 \
    --num_workers 4 


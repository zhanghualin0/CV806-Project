# #!/bin/bash



# Run the Python script with the necessary arguments
python ./tools/embs/save_blip_embs_imgs.py \
    --image_dir ./datasets/CIRR/images \
    --img_ext png \
    --batch_size 128 \
    --num_workers 4 \
    --model_type large





















#!/bin/bash

# # Base directory for images
# image_base_dir="./datasets/CIRR/images"
# # Base directory for embeddings
# embeddings_base_dir="./datasets/CIRR/blip-embs-large"

# # Process a directory ($1) and optionally a subdirectory ($2)
# process_dir() {
#     local image_dir="$1"
#     local subdir="$2"
#     local full_image_dir="$image_dir"
#     local full_embeddings_dir="$embeddings_base_dir"
    
#     if [[ -n "$subdir" ]]; then
#         full_image_dir="$image_dir/$subdir"
#         full_embeddings_dir="$embeddings_base_dir/$subdir"
#     fi
    
#     echo "Processing $full_image_dir..."
    
#     python ./tools/embs/save_blip_embs_imgs.py \
#         --image_dir "$full_image_dir" \
#         --save_dir "$full_embeddings_dir" \
#         --img_ext png \
#         --batch_size 128 \
#         --num_workers 4 \
#         --model_type large
# }

# # Process dev and test1 directly
# process_dir "$image_base_dir" "dev"
# process_dir "$image_base_dir" "test1"

# # Process each subdirectory within train
# for subdir in $(ls "$image_base_dir/train"); do
#     process_dir "$image_base_dir/train" "$subdir"
# done


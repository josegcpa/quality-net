python3 train-quality-net.py \
    --dataset_path dataset_hdf5/train.h5\
    --input_height 512 \
    --input_width 512 \
    --log_every_n_steps 2000 \
    --save_summary_folder summaries/quality-net \
    --save_checkpoint_folder checkpoints/quality-net/quality-net \
    --batch_size 32 \
    --number_of_epochs 25 \
    --learning_rate 0.00005 \
    --brightness_max_delta 0.02 \
    --saturation_lower 0.9 \
    --saturation_upper 1.1 \
    --hue_max_delta 0.1 \
    --contrast_lower 0.9 \
    --contrast_upper 1.1 \
    --min_jpeg_quality 90 \
    --max_jpeg_quality 100 \
    --probability 0.6

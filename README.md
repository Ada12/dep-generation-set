CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --out_dir=ce/no_tags/no_valid/1 \
    --dropout_p=0.1 \
    --epoch_max=501 \
    1>/dev/null 2>&1 &

python predict.py --out_dir=ce/tags/valid/1

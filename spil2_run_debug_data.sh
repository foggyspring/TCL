#!/bin/sh
# CUDA_VISIBLE_DEVICES=0 python spil/training.py \
# trainer.gpus=-1 \
# datamodule.root_data_dir=/home/ubuntu/dataset/calvin/calvin_debug_dataset \
# #datamodule/datasets=vision_lang_shm \
# datamodule/datasets=vision_only\
# model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR_ABC'


#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python spil/training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=/home/ubuntu/dataset/calvin/calvin_debug_dataset \
model=spil \
model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR_D' \
callbacks/checkpoint=lh_sr \
trainer.max_epochs=50 \
datamodule/datasets=vision_lang \
datamodule.datasets.lang_dataset.batch_size=32 \
datamodule.datasets.vision_dataset.batch_size=32 \

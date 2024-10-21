#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=/fast_data/calvin/task_ABC_D \
datamodule/datasets=vision_lang_shm \
datamodule.datasets.lang_dataset.batch_size=32 \
datamodule.datasets.vision_dataset.batch_size=32 \
model.action_decoder.sg_chk_path='../checkpoints/SKILL_GENERATOR_ABC'
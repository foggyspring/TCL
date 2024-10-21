# #!/bin/sh
# CUDA_VISIBLE_DEVICES=0 python spil/training.py 
# trainer.gpus=-1 \
# datamodule.root_data_dir=/home/ubuntu/dataset/calvin/task_D_D \
# #datamodule/datasets=vision_lang_shm \
# datamodule/datasets=lang_only \
# model=hulc \
# #datamodule.datasets.lang_dataset.batch_size=64 \
# #datamodule.datasets.vision_dataset.batch_size=64 \
# #datamodule=default \
# trainer.max_epochs=50 \
# loss=hulc \

CUDA_VISIBLE_DEVICES=0 python spil/training.py 
trainer.gpus=-1 \
datamodule.root_data_dir=/home/ubuntu/dataset/calvin/task_ABC_D \
datamodule/datasets=lang_only \
#datamodule/datasets=vision_lang_shm \
#datamodule.datasets.lang_dataset.batch_size=64 \
#datamodule.datasets.vision_dataset.batch_size=64 \
model=hulc \
#datamodule=default \
loss=hulc \
trainer.max_epochs=50 \

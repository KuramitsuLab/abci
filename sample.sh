#!/bin/bash

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/apps/centos7/python/3.8.7/lib

python3 train_mt5.py\
    --name='mt5_35'\
    --masking --masking_ratio=0.35\
    --limit_batches=1000\
    --model_name_or_path='google/mt5-small'\
    --tokenizer_name_or_path='google/mt5-small'\
    --additional_tokens=''\
    --save_checkpoint\
    --num_workers=4\
    --batch_size=8\
    python_corpus/*.txt
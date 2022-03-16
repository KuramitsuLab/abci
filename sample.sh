#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=24:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/apps/centos7/python/3.8.7/lib

python3 train_mlm.py\
    --name='mg_40'\
    --masking --masking_ratio=0.40\
    --max_epochs=3\
    --model_name_or_path='megagonlabs/t5-base-japanese-web'\
    --tokenizer_name_or_path='megagonlabs/t5-base-japanese-web'\
    --additional_tokens=''\
    --save_checkpoint\
    --num_workers=4\
    --batch_size=0\
    python_corpus/*.txt
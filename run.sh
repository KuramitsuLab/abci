#!/bin/bash


#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd


source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/9.2/9.2.88.1

#!/bin/bash -e
pip3 install --user --upgrade pip
pip3 install -r requirements.txt
python3 main_conala.py conala_mined.tsv

#!/bin/bash
# ./a.out
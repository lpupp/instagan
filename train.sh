#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega
DATA_PATH=${BASE_PATH}/data/InstaGAN

mkdir ${BASE_PATH}/scratch/InstaGAN
mkdir ${BASE_PATH}/scratch/InstaGAN/checkpoints

source ~/tensorflow/bin/activate

srun python instagan/train.py --dataroot ${DATA_PATH}/fashion/shoes2dresses/ --checkpoints_dir  ${BASE_PATH}/scratch/InstaGAN/checkpoints --no_flip --resize_or_crop none --model insta_gan --name instagan_128 --loadSizeH 128 --loadSizeW 128 --fineSizeH 128 --fineSizeW 128 --niter 10 --niter_decay 70 >> ${BASE_PATH}/scratch/InstaGAN/instagan_128.txt

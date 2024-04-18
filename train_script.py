import subprocess
import os

if __name__ == '__main__':

    python_path = 'train_DDP_A_valid_B_0418.py'
    config_path = 'configs/config_0418_Xpresso.yaml'
    
    subprocess.run(
        f'export OMP_NUM_THREADS=4 ;'
        f'export CUDA_VISIBLE_DEVICES=0,1,2,3 ;'
        f'export MASTER_PORT=23456 ;'
        f'torchrun --nproc_per_node=4 {python_path} --config_path {config_path}', 
        shell=True)

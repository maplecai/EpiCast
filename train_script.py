import subprocess
import os

if __name__ == '__main__':

    python_path = 'train_0424.py'
    config_path = 'configs/config_0426_ATAC.yaml'
    
    subprocess.run(
        f'export OMP_NUM_THREADS=4 ;'
        f'export CUDA_VISIBLE_DEVICES=3 ;'
        f'export MASTER_PORT=14285 ;'
        f'torchrun --nproc_per_node=1 {python_path} --config_path {config_path}', 
        shell=True)

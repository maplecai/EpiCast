import subprocess
import os
import yaml

if __name__ == '__main__':

    python_path = 'train_0506.py'
    config_path = 'configs/config_0506_cls_23.yaml'

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if config['distribute'] == False:
        subprocess.run(f'python {python_path} --config_path {config_path}', shell=True)
    
    else:
        subprocess.run(
            f'export OMP_NUM_THREADS=4 ;'
            f'export CUDA_VISIBLE_DEVICES=0,1 ;'
            f'export MASTER_PORT=14285 ;'
            f'torchrun --nproc_per_node=2 {python_path} --config_path {config_path}', 
            shell=True)

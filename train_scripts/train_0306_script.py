import os
import subprocess
from ruamel.yaml import YAML

yaml = YAML()

if __name__ == '__main__':

    script_path = 'train_scripts/train_0306.py'
    config_path = 'configs/config_0307_SirajMPRA_3_cell_types.yaml'

    subprocess.run(
        f'OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 {script_path} -c {config_path}',
        shell=True)


    # # for cell_type in ['K562', 'HepG2', 'SK-N-SH', 'A549', 'HCT116']:
    # for cell_type in ['A549', 'HCT116']:
    #     new_config_path = f'{config_path[:-5]}_without_{cell_type}.yaml'
            
    #     with open(config_path, 'r') as f:
    #         config = yaml.load(f)
        
    #     config['cell_types'].remove(cell_type)
    #     config['model']['args']['num_cell_types'] = 4

    #     with open(new_config_path, 'w') as f:
    #         yaml.dump(config, f)

    #     subprocess.run(
    #         f'OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 {script_path} -c {new_config_path};',
    #         shell=True)

    #     # subprocess.run(
    #     #     f'python {script_path} --config_path {new_config_path};',
    #     #     shell=True)

import os
import subprocess
from ruamel.yaml import YAML

yaml = YAML()

if __name__ == '__main__':

    script_path = 'train_scripts/train_0306.py'
    config_path = 'configs/config_0318_SirajMPRA_four_cell_type.yaml'

    # subprocess.run(
    #     f'OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 {script_path} -c {config_path}',
    #     shell=True)

    for cell_type in ['K562', 'HepG2', 'SK-N-SH', 'A549', 'HCT116']:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        
        config['cell_types'].remove(cell_type)
        config['num_cell_types'] = 4
        # config['cell_types'][0] = cell_type # should not change config['cell_types'] list, because yaml anchor and ref will change

        new_config_path = f'{config_path.split(".")[0]}_without_{cell_type}.yaml'
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)

        subprocess.run(
            f'OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=2 {script_path} -c {new_config_path};',
            # f'python {script_path} --config_path {new_config_path};',
            shell=True)

import subprocess
import yaml
import os

def change_config(load_dir, save_dir, key, value):
    with open(load_dir, 'r') as f:
        # config = yaml.safe_load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
        config[key] = value

    if save_dir is not None:
        with open(save_dir, 'w') as f:
            yaml.dump(config, f)


if __name__ == '__main__':

    python_path = 'train_0424.py'
    config_path = 'configs/config_0425_cls.yaml'

    value_list = [[0,1,2], [3], [1], [2], [1,3], [2,3], [0,1,2,3],]

    for value in value_list:
        # value_str = ''.join(str(i) for i in value)
        # save_dir = config_path.replace('.yaml', f'_{value_str}.yaml')
        change_config(config_path, config_path, 'selected_train_datasets_idx', value)
        change_config(config_path, config_path, 'selected_valid_datasets_idx', value)

        subprocess.run(
            f'export OMP_NUM_THREADS=4 ;'
            f'export CUDA_VISIBLE_DEVICES=3 ;'
            f'torchrun --nproc_per_node=1 {python_path} --config_path {config_path};',
            shell=True)





    # saved_dir = 'saved/0425_cls'
    # folder_list = os.listdir(saved_dir)

    # for folder in folder_list:
    #     python_path = 'train_0424_valid.py'
    #     config_path = f'{saved_dir}/{folder}/config.yaml'

    #     config_path_valid = config_path.replace('config.yaml', 'config_valid.yaml')

    #     change_config(config_path, config_path_valid, 'selected_valid_datasets_idx', [3])

    #     subprocess.run(
    #         f'export OMP_NUM_THREADS=4 ;'
    #         f'export CUDA_VISIBLE_DEVICES=3 ;'
    #         f'torchrun --nproc_per_node=1 {python_path} --config_path {config_path_valid};',
    #         shell=True)
        
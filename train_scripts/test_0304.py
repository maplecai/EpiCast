import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *
from train_0306 import Trainer


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '-s', '--saved_dir', type=str, default=None, 
        help='saved folder dir',
        )
    args = args.parse_args()

    config_path = 'configs/config_0306_SirajMPRA_test.yaml'
    config = utils.load_config(config_path)

    saved_dir = args.saved_dir
    config_path = os.path.join(saved_dir, 'config.yaml')
    config = utils.load_config(config_path)
    config['train'] = False
    config['distributed'] = False
    config['gpu_ids'] = [3]
    config['load_saved_model'] = True
    config['saved_model_path'] = os.path.join(saved_dir, 'checkpoint.pth')
    config['cell_types'] = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    config['test_dataset']['args']['cell_types'] = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    # 之后需要研究一下这两行能不能用yaml引用合并

    trainer = Trainer(config)

    test_dataset = utils.init_obj(datasets, config['test_dataset'])
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    trainer.test(test_loader)
    torch.cuda.empty_cache()

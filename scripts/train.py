import argparse
from pathlib import Path
from epicast import models, datasets, metrics, utils, training

ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config_path = args.config_path

    config = utils.load_config(config_path)
    # config = utils.resolve_config_paths(config, ROOT_DIR)
    config = utils.process_config(config)

    trainer = utils.init_obj(
        training, 
        config['trainer'],
        config
    )

    trainer.train()
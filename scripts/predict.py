import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from omegaconf import OmegaConf
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from epicast import datasets, models, utils  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EpiCast inference only.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/predict.py saved/exp/run_xxx/config.yaml\n\n"
            "  python scripts/predict.py saved/exp/run_xxx/config.yaml \\\n"
            "    --pred_name test_pred.npy \\\n"
            "    --reverse_comp \\\n"
            "    total_dataset.args.epi_file_path=data/new_vef.tsv\n"
        ),
    )
    parser.add_argument("--config_path", type=str, required=True, help="config yaml path")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pred_name", type=str, default="pred.npy")
    parser.add_argument("--reverse_comp", action="store_true")
    args, unknown = parser.parse_known_args()
    return args, unknown


def resolve_device(device):
    if device != "auto":
        return device
    free_gpus = utils.get_free_gpus()
    if free_gpus:
        return free_gpus[0]
    return "cpu"


def load_config(args, unknown):
    config_path = Path(args.config_path)
    cfg = OmegaConf.load(str(config_path))
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return cfg, config_path


def load_model(cfg, ckpt_path, device):
    model = utils.init_obj(models, cfg["model"])
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_loader(cfg):
    dataset = utils.init_obj(datasets, cfg["total_dataset"])
    loader = utils.init_obj(torch.utils.data, cfg["val_loader"], dataset=dataset, sampler=None)
    return loader


@torch.no_grad()
def infer(model, loader, device, reverse_comp):
    pred_batches = []
    for batch in tqdm(loader):
        batch = utils.to_device(batch, device)
        if not reverse_comp:
            pred = model(batch)
        else:
            pred1 = model(batch)
            batch["seq"] = batch["seq"].flip(dims=[1, 2])
            pred2 = model(batch)
            pred = (pred1 + pred2) / 2
        pred_batches.append(pred.detach().cpu())
    return torch.cat(pred_batches, dim=0).numpy()


def main():
    args, unknown = parse_args()
    cfg, config_path = load_config(args, unknown)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else config_path.parent / "checkpoints" / "best.pth"
    )
    model = load_model(cfg, ckpt_path, device)
    loader = build_loader(cfg)

    pred_arr = infer(model, loader, device, args.reverse_comp)

    out_dir = Path(cfg["saved_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / args.pred_name
    np.save(pred_path, pred_arr)

    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"saved {pred_path} shape={pred_arr.shape}")


if __name__ == "__main__":
    main()

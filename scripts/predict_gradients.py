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


SKIP_GRAD_KEYS = {"label", "labels", "target", "targets", "y", "idx", "index"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EpiCast prediction and input gradients.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/predict_gradients.py \\\n"
            "    --saved_dir saved/exp/run_xxx\n\n"
            "  python scripts/predict_gradients.py \\\n"
            "    --saved_dir saved/exp/run_xxx \\\n"
            "    --dataset_key test_dataset \\\n"
            "    --grad_input_keys seq,VEF \\\n"
            "    --grad_output_dim 0 \\\n"
            "    test_dataset.args.filter_in_list=[chr7,chr13] \\\n"
            "    val_loader.args.batch_size=512\n\n"
            "Smoke check:\n"
            "  1) confirm files: pred.npy, grad_<key>.npy\n"
            "  2) np.load each file and verify first dim equals dataset size"
        ),
    )
    parser.add_argument("--saved_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--pred_name", type=str, default="pred.npy")
    parser.add_argument("--dataset_key", type=str, default="total_dataset")
    parser.add_argument("--loader_key", type=str, default="val_loader")
    parser.add_argument("--grad_input_keys", type=str, default=None)
    parser.add_argument("--grad_output_dim", type=int, default=None)
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
    saved_dir = Path(args.saved_dir)
    config_path = Path(args.config) if args.config else saved_dir / "config.yaml"
    cfg = OmegaConf.load(str(config_path))
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def load_model(cfg, args, device):
    model = utils.init_obj(models, cfg["model"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.saved_dir) / "checkpoint.pth"
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_loader(cfg, dataset_key, loader_key):
    dataset = utils.init_obj(datasets, cfg[dataset_key])
    if loader_key in cfg:
        return utils.init_obj(torch.utils.data, cfg[loader_key], dataset=dataset, sampler=None)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )


def pick_input_keys(batch, user_keys):
    if user_keys:
        return [k for k in user_keys if k in batch]
    keys = []
    for k, v in batch.items():
        if not torch.is_tensor(v):
            continue
        if not v.is_floating_point():
            continue
        if k.lower() in SKIP_GRAD_KEYS:
            continue
        keys.append(k)
    return keys


def select_grad_target(pred, output_dim):
    if output_dim is None:
        return pred.sum()
    if pred.ndim == 1:
        return pred.sum()
    return pred[:, output_dim].sum()


def run_pred_and_grad(model, loader, device, grad_input_keys, grad_output_dim):
    pred_batches = []
    grad_storage = {}

    for batch in tqdm(loader):
        batch = utils.to_device(batch, device)
        input_keys = pick_input_keys(batch, grad_input_keys)
        for k in input_keys:
            batch[k] = batch[k].detach().clone().requires_grad_(True)

        pred = model(batch)
        pred_batches.append(pred.detach().cpu())

        target = select_grad_target(pred, grad_output_dim)
        model.zero_grad()
        target.backward()

        for k in input_keys:
            grad_storage.setdefault(k, []).append(batch[k].grad.detach().cpu())

    pred_arr = torch.cat(pred_batches, dim=0).numpy()
    grad_arr = {k: torch.cat(v, dim=0).numpy() for k, v in grad_storage.items()}
    return pred_arr, grad_arr


def main():
    args, unknown = parse_args()
    device = resolve_device(args.device)
    cfg = load_config(args, unknown)
    model = load_model(cfg, args, device)
    loader = build_loader(cfg, args.dataset_key, args.loader_key)

    grad_input_keys = None
    if args.grad_input_keys:
        grad_input_keys = [x.strip() for x in args.grad_input_keys.split(",") if x.strip()]

    pred_arr, grad_arr = run_pred_and_grad(
        model=model,
        loader=loader,
        device=device,
        grad_input_keys=grad_input_keys,
        grad_output_dim=args.grad_output_dim,
    )

    out_dir = Path(args.save_dir) if args.save_dir else Path(args.saved_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / args.pred_name
    np.save(pred_path, pred_arr)
    print(f"device={device}")
    print(f"saved {pred_path} shape={pred_arr.shape}")

    for k, v in grad_arr.items():
        grad_path = out_dir / f"grad_{k}.npy"
        np.save(grad_path, v)
        print(f"saved {grad_path} shape={v.shape}")


if __name__ == "__main__":
    main()

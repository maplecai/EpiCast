import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils


@torch.no_grad()
def run_inference(model, test_loader, device, output_path):
    model.eval()
    model = model.to(device)
    pred_list = []
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch = utils.to_device(batch, device)
        pred = model(batch)
        pred_list.append(pred.detach())

    pred_list = torch.cat(pred_list).cpu().numpy()
    np.save(output_path, pred_list)
    torch.cuda.empty_cache()
    return


def run_compute_all_gradients(model, test_loader, device, out_dir):
    """
    Automatically compute gradients dy/dx for ALL float input tensors in batch.
    Saves grad_KEY.npy for every input key.
    """
    print(f"Saving gradients to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    model.to(device)

    grad_dict = {}   # key → list of batches of grads

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        # move fields to device
        for k in batch:
            batch[k] = batch[k].to(device)

        # discover input tensor keys
        input_keys = []
        for k, v in batch.items():
            if not torch.is_tensor(v):
                continue
            if v.dtype not in (torch.float32, torch.float64):
                continue
            # skip label/target/index if any (heuristic)
            if k.lower() in ["label", "labels", "target", "targets", "y"]:
                continue

            batch[k] = v.clone().detach().requires_grad_(True)
            input_keys.append(k)

        # forward
        pred = model(batch)

        # scalar for backward
        loss = pred.sum()

        # backward
        model.zero_grad()
        loss.backward()

        # collect grads
        for k in input_keys:
            grad_val = batch[k].grad.detach().cpu()

            if k not in grad_dict:
                grad_dict[k] = []
            grad_dict[k].append(grad_val)

    # concat & save
    for k, v in grad_dict.items():
        arr = torch.cat(v, dim=0).numpy()   # shape: (N, ...)
        save_path = os.path.join(out_dir, f"grad_{k}.npy")
        np.save(save_path, arr)
        print(f"Saved {save_path} | shape = {arr.shape}")

    torch.cuda.empty_cache()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--saved_dir', type=str, default=None)
    parser.add_argument('-c', '--config_path', type=str, default=None)
    parser.add_argument('-de', '--device', type=str, default='cuda:0')
    parser.add_argument('-o', '--output_name', type=str, default='pred.npy')

    parser.add_argument('-d', '--data_path', type=str, default=None)
    parser.add_argument('--seq_file_path', type=str, default=None)
    parser.add_argument('--epi_file_path', type=str, default=None)

    parser.add_argument('--compute_grad', action='store_true',
                        help="Compute gradients for ALL input fields")

    args = parser.parse_args()

    saved_dir = args.saved_dir
    config_path = args.config_path
    device = args.device
    output_name = args.output_name

    data_path = args.data_path
    seq_file_path = args.seq_file_path
    epi_file_path = args.epi_file_path

    # load config
    if config_path is None:
        config_path = os.path.join(saved_dir, 'config.yaml')
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')

    config = utils.load_config(config_path)

    # override dataset paths if provided
    if data_path is not None:
        config['total_dataset']['args']['data_path'] = data_path
    if seq_file_path is not None:
        config['total_dataset']['args']['seq_file_path'] = seq_file_path
    if epi_file_path is not None:
        config['total_dataset']['args']['epi_file_path'] = epi_file_path

    # load model
    model = utils.init_obj(models, config['model'])
    saved_model_path = os.path.join(saved_dir, 'checkpoint.pth')
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)

    # data loader
    total_dataset = utils.init_obj(datasets, config['total_dataset'])
    total_loader = DataLoader(
        dataset=total_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # ---------- select mode ----------
    if args.compute_grad:
        print("==> Computing gradients for ALL input fields...")

        out_dir = os.path.join(saved_dir, args.output_name)
        run_compute_all_gradients(model, total_loader, device, out_dir)

    else:
        print("==> Running inference...")
        output_path = os.path.join(saved_dir, output_name)
        run_inference(model, total_loader, device, output_path)


if __name__ == '__main__':
    main()

import sys
sys.path.append("..")
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

# sys.path.append('../pretrained_models/enformer-pytorch')
# from enformer_pytorch import from_pretrained

from MPRA_predict.models.enformer_pytorch import from_pretrained as Enformer_from_pretrained


def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch['seq']
            x = x.to(device)
            output = model(x)
            # if enformer
            if isinstance(output, dict):
                output = output['human']
            y_pred.append(output.detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    torch.cuda.empty_cache()
    return y_pred



# split to many parts, predict and save, in order to save memory
def get_pred_split(model, dataset, device, num_splits):

    split_size = len(dataset) // num_splits  # num_splits是你要分割的部分数
    # 分割数据集
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(dataset)

        subset = Subset(dataset, range(start_idx, end_idx))
        subloader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
        y_pred = get_pred(model, subloader, device)
        np.save(f'{output_path}_{i}.npy', np.array(y_pred))
    return


if __name__ == "__main__":
    
    set_seed(0)
    model_path = f'../pretrained_models/enformer_weights'
    data_path = f'/home/hxcai/CRE/MPRA_predict/data/SirajMPRA/SirajMPRA_562654.csv'
    output_path = f'outputs/SirajMPRA_Enformer_no_padding.npy'

    if os.path.exists(output_path):
        print(f'already have {output_path}')
        sys.exit()
    else:
        print(f'predicting {output_path}')

    model = Enformer_from_pretrained(model_path, use_tf_gamma=False, target_length=2)
    # torchinfo.summary(model, input_size=(1, 256, 4), depth=5)

    # no padding
    
    dataset = SeqDataset(
        data_path=data_path,
        input_column='seq', 
        crop=False, ###
        padding=False, ###
        N_fill_value=0
        )
    
    get_pred_split(model, dataset, 'cuda:0', 10)
    
    # test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    # pred = get_pred(model, test_data_loader)
    # np.save(output_path, pred)

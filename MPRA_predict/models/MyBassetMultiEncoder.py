from .MyBasset import *
from .. import models
from .. import utils
import yaml


class MyBassetMultiEncoderMultiDecoder(nn.Module):
    def __init__(
        self, 
        input_length=200,
        output_dim=1,
        n_cells=2,
        n_tasks=2,
        **kwargs,
    ):
        super().__init__()

        self.encoder_list = nn.ModuleList([
            utils.init_obj(models, kwargs['encoder_kwargs']) for i in range(n_cells)])

        with torch.no_grad():
            x = torch.randn(2, 4, input_length)
            x = self.encoder_list[0](x)
            x = x.view(x.size(0), -1)
            hidden_dim = x.shape[1]

        self.processor = utils.init_obj(models, kwargs['processor_kwargs'], input_dim=hidden_dim)

        with torch.no_grad():
            x = self.processor(x)
            x = x.view(x.size(0), -1)
            hidden_dim = x.shape[1]

        self.decoder_list = nn.ModuleList([
            utils.init_obj(models, kwargs.get('decoder_kwargs', {}), input_dim=hidden_dim, output_dim=output_dim) for i in range(n_tasks)])
        

    def forward(self, x, cell_idx, task_idx):
        if x.shape[2] == 4:
            x = x.transpose(1, 2)
        # 假设一个batch内只有一种类型的数据
        if cell_idx.dim() > 0:
            cell_idx = cell_idx[0].long()
        if task_idx.dim() > 0:
            task_idx = task_idx[0].long()

        x = self.encoder_list[cell_idx](x)
        x = x.view(x.size(0), -1)
        x = self.processor(x)
        x = x.view(x.size(0), -1)
        x = self.decoder_list[task_idx](x)
        # x = x.squeeze(-1)

        return x





if __name__ == '__main__':


    yaml_str = '''

model:
    type: MyBassetMultiEncoderMultiDecoder
    args:
        input_length:   200
        output_dim: 1
        n_cells:    2
        n_tasks:    2

        encoder_kwargs:
            type: MyBassetEncoder
            args:
                conv_channels_list:     [256,256,256]
                conv_kernel_size_list:  [7,7,7]
                conv_padding_list:      [3,3,3]
                pool_kernel_size_list:  [2,2,2]
                pool_padding_list:      [0,0,0]
                conv_dropout_rate:      0.2
        
        processor_kwargs:
            type: MyBassetDecoder
            args:
                linear_channels_list:   [256, 256]
                linear_dropout_rate:    0.5
                last_linear_layer:      false
                sigmoid:                false
    
        decoder_kwargs:
            type: MyBassetDecoder
            args:
                linear_channels_list:   []
                linear_dropout_rate:    0.5
                last_linear_layer:      true
                sigmoid:                true

        # augmentation:           false
        # augmentation_region:    null


    '''

    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = utils.init_obj(models, config['model'])

    x = torch.randn(2, 200, 4)
    t = torch.tensor([0, 1])
    output = model(x, t, t)
    torchinfo.summary(model, input_data=(x, t, t), depth=4)

import torch
import torch.nn as nn

from .conv_block import ConvBlock, ResConvBlock
from .linear_block import LinearBlock, SqueezeLayer
from .trans_block import TransBlock
from .film import FiLM

class ResTransformerV2(nn.Module):
    def __init__(
        self, 
        input_length=196608,
        input_channels=4,
        output_dim=1,
        target_length=None,
        last_activation=None,
        squeeze=True,

        num_conv_blocks=6,
        conv_channels=256,
        conv_kernel_size=3,
        conv_padding='same',
        conv_dropout_rate=0.2,
        pool_kernel_size=2,
        conv_layer_order='resnet_v2',
        conv_activation='relu',

        num_trans_blocks=3, 
        trans_d_embed=256, 
        trans_n_heads=8, 
        trans_d_mlp=256,
        trans_dropout_rate=0.1,

        trans_add_cls=False,
        trans_output_mode='seq_all',
        
        fusion_type = 'film',

        num_linear_blocks=0,
        linear_channels=1024,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_length       = input_length
        self.input_channels     = input_channels
        self.output_dim         = output_dim
        self.total_token_length = self.input_length // (pool_kernel_size ** num_conv_blocks) # 1536 or 1024
        self.target_length      = target_length # 896
        self.squeeze            = squeeze
        self.fusion_type        = fusion_type


        self.trans_output_mode  = trans_output_mode
        self.trans_add_cls      = trans_add_cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))

        self.conv_layers = nn.Sequential()
        for i in range(num_conv_blocks):
            self.conv_layers.add_module(
                f'conv_block_{i}', ResConvBlock(
                    in_channels=input_channels if i == 0 else conv_channels, 
                    out_channels=conv_channels, 
                    kernel_size=conv_kernel_size, 
                    stride=1, 
                    padding=conv_padding,
                    activation=conv_activation,
                    layer_order=conv_layer_order,
                )
            )

            if pool_kernel_size != 1:
                self.conv_layers.add_module(
                    f'max_pool_{i}', nn.MaxPool1d(
                        kernel_size=pool_kernel_size, 
                        ceil_mode=True, # keep edge information
                    )
                )
            self.conv_layers.add_module(
                f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate)
            )


        self.trans_layers = nn.Sequential()
        for i in range(num_trans_blocks):
            self.trans_layers.add_module(
                f'transformer_block_{i}', TransBlock(
                    d_embed=trans_d_embed, 
                    n_heads=trans_n_heads, 
                    d_mlp=trans_d_mlp, 
                    dropout_rate=trans_dropout_rate
                )
            )

        
        self.linear_layers = nn.Sequential()
        for i in range(num_linear_blocks):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=trans_d_embed, 
                    out_channels=linear_channels,
                    activation='relu',
                )
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=linear_channels, 
                out_features=output_dim,
            )
        )

        if last_activation is None:
            pass
        elif last_activation == 'sigmoid':
            self.linear_layers.add_module(
                f'sigmoid_layer', nn.Sigmoid()
            )
        elif last_activation == 'softmax':
            self.linear_layers.add_module(
                f'softmax_layer', nn.Softmax(dim=-1)
            )
        elif last_activation == 'softplus':
            self.linear_layers.add_module(
                f'softplus_layer', nn.Softplus()
            )
        else:
            raise ValueError(f"Unsupported last_activation mode: {self.last_activation}")
    
        # if squeeze is True:
        #     self.linear_layers.add_module(
        #         f'squeeze_layer', SqueezeLayer()
        #     )


    def forward_trans_layers(self, tokens: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, hidden_dim = tokens.shape

        if self.trans_add_cls:
            tokens = torch.cat([self.cls_token.expand(batch_size, 1, -1), tokens], 1)
        cls_len = int(self.trans_add_cls)

        out = self.trans_layers(tokens)

        if self.trans_output_mode == 'cls':
            return out[:, 0]
        elif self.trans_output_mode == 'seq_mean':
            return out[:, cls_len:].mean(1)
        elif self.trans_output_mode == 'seq_all':
            return out[:, cls_len:]
        elif self.trans_output_mode == 'all':
            return out
        else:
            raise ValueError(f"Unsupported trans_output_mode mode: {self.trans_output_mode}")


    def forward(self, inputs: torch.Tensor | dict | list) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            seq = inputs
        elif isinstance(inputs, dict):
            seq = inputs.get('seq', None)
        elif isinstance(inputs, list):
            seq = inputs[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        batch_size = seq.shape[0]
        expected_shape = (batch_size, self.input_length, self.input_channels)
        assert seq.shape == expected_shape, f"{seq.shape = }, {expected_shape = }"

        seq = seq.permute(0, 2, 1)
        seq_embedding = self.conv_layers(seq)
        seq_embedding = seq_embedding.permute(0, 2, 1)
        out = self.forward_trans_layers(seq_embedding)

        if self.target_length is not None:
            start = (self.total_token_length - self.target_length) // 2
            end = start + self.target_length
            out = out[:, start:end]
        

        out = self.linear_layers(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out



if __name__ == '__main__':

    yaml_str = '''
        model:
            _target_: varlen_genomics.models.MyResTransformer

            input_length:       200
            input_channels:     4
            output_dim:             1
            sigmoid:                False
            squeeze:                True

            conv_first_channels:    256
            conv_first_kernel_size: 7
            conv_layer_order:       conv_bn_add_relu
            conv_channels_list:     [256,256,256,256,256,256]
            conv_kernel_size_list:  [3,3,3,3,3,3]
            pool_kernel_size_list:  [2,2,2,2,2,2]
            conv_dropout_rate:      0.2

            num_trans_blocks: 3
            trans_d_embed: 256
            trans_n_heads: 4
            trans_d_mlp: 256
            trans_dropout_rate: 0.2
            trans_add_cls: False
            trans_output_mode: seq_all

            linear_channels_list: [1024]
            linear_dropout_rate: 0.5
        '''
    import yaml
    import torchinfo
    from hydra.utils import instantiate

    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = instantiate(config['model'])

    seq = torch.zeros(size=(16, 4, 200))
    inputs = {'seq': seq}

    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )
    
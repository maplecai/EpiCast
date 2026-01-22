import torch
import torch.nn as nn
import torchinfo
from collections import OrderedDict

from .conv_block import ConvBlock, ResConvBlock
from .linear_block import LinearBlock
from .trans_block import TransBlock


class ResTransformer(nn.Module):
    def __init__(
        self, 
        input_length=200,
        input_channels=4,
        output_dim=1,

        last_activation=None,
        sigmoid=False, # deprecated
        squeeze=True,

        conv_first_channels=256,
        conv_first_kernel_size=7,
        pool_first_kernel_size=1,

        conv_padding='same',
        conv_activation='relu',
        conv_layer_order='conv_bn_add_relu',
        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_dropout_rate=0.2,
        pool_kernel_size_list=None,

        num_trans_blocks=3, 
        trans_d_embed=256, 
        trans_n_heads=8, 
        trans_d_mlp=256,
        trans_dropout_rate=0.1,

        trans_output_mode='seq_mean',
        trans_add_cls=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_length   = input_length
        self.input_channels = input_channels
        self.output_dim         = output_dim
        self.last_activation    = last_activation
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze

        self.trans_output_mode       = trans_output_mode
        self.trans_add_cls      = trans_add_cls

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))

        if linear_channels_list is None:
            linear_channels_list = []

        self.conv_layers = nn.Sequential(OrderedDict([]))
        
        if conv_first_channels is not None:
            self.conv_layers.add_module(
                f'conv_block_first', ConvBlock(
                    in_channels=input_channels,
                    out_channels=conv_first_channels, 
                    kernel_size=conv_first_kernel_size, 
                    stride=1,
                    padding=conv_padding,
                    layer_order='conv_relu_bn',
                    activation=conv_activation,
                )
            )
        
        if pool_first_kernel_size != 1:
            self.conv_layers.add_module(
                f'max_pool_first', nn.MaxPool1d(
                    kernel_size=pool_first_kernel_size, 
                    ceil_mode=True, # keep edge information
                )
            )

        if conv_channels_list is not None:
            for i in range(len(conv_channels_list)):
                self.conv_layers.add_module(
                    f'res_conv_block_{i}', ResConvBlock(
                        in_channels=conv_first_channels if i == 0 else conv_channels_list[i-1], 
                        out_channels=conv_channels_list[i], 
                        kernel_size=conv_kernel_size_list[i], 
                        stride=1, 
                        padding=conv_padding,
                        layer_order=conv_layer_order,
                        activation=conv_activation,
                    )
                )

                if pool_kernel_size_list[i] != 1:
                    self.conv_layers.add_module(
                        f'max_pool_{i}', nn.MaxPool1d(
                            kernel_size=pool_kernel_size_list[i], 
                            ceil_mode=True, # keep edge information
                        )
                    )
                self.conv_layers.add_module(
                    f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate)
                )

            if conv_channels_list[-1] != trans_d_embed:
                self.conv_layers.add_module(
                    f'conv_reshape', nn.Conv1d(
                        in_channels=conv_channels_list[-1], 
                        out_channels=trans_d_embed, 
                        kernel_size=1
                    )
                )

        self.trans_layers = nn.Sequential(OrderedDict([]))
        if num_trans_blocks is not None:
            for i in range(num_trans_blocks):
                self.trans_layers.add_module(
                    f'transformer_block_{i}', TransBlock(
                        d_embed=trans_d_embed, 
                        n_heads=trans_n_heads, 
                        d_mlp=trans_d_mlp, 
                        dropout_rate=trans_dropout_rate
                    )
                )

        # compute the shape
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_length, self.input_channels)
            dummy = dummy.permute(0, 2, 1)
            dummy = self.conv_layers(dummy) # (batch_size, conv_channels, seq_length)
            dummy = dummy.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)
            dummy = self.trans_layers(dummy) # (batch_size, seq_length, hidden_dim)
            dummy = dummy.mean(1) # (batch_size, hidden_dim)
        current_dim = dummy.shape[1]


        
        self.linear_layers = nn.Sequential(OrderedDict([]))
        if linear_channels_list is not None:
            for i in range(len(linear_channels_list)):
                self.linear_layers.add_module(
                    f'linear_block_{i}', LinearBlock(
                        in_channels=current_dim, 
                        out_channels=linear_channels_list[i],
                        activation='relu',
                    )
                )
                self.linear_layers.add_module(
                    f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
                )
                current_dim = linear_channels_list[i]

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=current_dim, 
                out_features=output_dim,
            )
        )

        if self.last_activation is not None:
            if self.last_activation == 'sigmoid':
                self.linear_layers.add_module(
                    f'sigmoid_layer', nn.Sigmoid()
                )
            elif self.last_activation == 'softmax':
                self.linear_layers.add_module(
                    f'softmax_layer', nn.Softmax(dim=-1)
                )
            elif self.last_activation == 'softplus':
                self.linear_layers.add_module(
                    f'softplus_layer', nn.Softplus()
                )
            else:
                raise ValueError(f"Unsupported last_activation mode: {self.last_activation}")




    def forward_trans_layers(self, tokens: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, hidden_dim = tokens.shape

        if self.trans_add_cls is False:
            out = self.trans_layers(tokens)
            if self.trans_output_mode == 'cls':
                raise ValueError(f"{self.trans_add_cls = }, but {self.trans_output_mode = }")
            elif self.trans_output_mode == 'seq_mean':
                out = out.mean(1)
            elif self.trans_output_mode == 'seq_all':
                out = out
            else:
                raise ValueError(f"Unsupported trans_output_mode mode: {self.trans_output_mode}")

        else:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            out = torch.cat([cls_token, tokens], dim=1)
            out = self.trans_layers(out)
            if self.trans_output_mode == 'cls':
                out = out[:, 0]
            elif self.trans_output_mode == 'seq_mean':
                out = out[:, 1:].mean(1)
            elif self.trans_output_mode == 'seq_all':
                out = out[:, 1:]
            else:
                raise ValueError(f"Unsupported trans_output_mode mode: {self.trans_output_mode}")
        return out


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
    
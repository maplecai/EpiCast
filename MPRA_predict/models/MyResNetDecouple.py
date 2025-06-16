import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils
from .MyResNet import MyResNet #, LinearBlock
from .ConvBlock import ConvBlock, ResConvBlock
from .MLP import MLP





class ZeroMask(nn.Module):
    """
    随机置 0 不做任何幅度补偿。
    """
    def __init__(self, p: float):
        """
        p: 每个元素被置 0 的概率
        """
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in the interval [0, 1).")
        self.p = p

    def forward(self, x: torch.Tensor):
        if self.training and self.p > 0.0:
            mask = torch.rand_like(x) > self.p
            x = x * mask
        else:
            x = x
        return x      # 不做 1/(1-p) 放大






# class MyResNetDecouple(nn.Module):
#     def __init__(
#         self, 
#         input_seq_length=200,
#         input_seq_channels=4,
#         input_feature_dim=0,
#         output_dim=1,

#         sigmoid=False,
#         squeeze=True,

#         conv_first_channels=256,
#         conv_first_kernel_size=7,
#         pool_first_kernel_size=1,

#         conv_padding='same',
#         conv_activation='relu',
#         conv_layer_order='conv_bn_add_relu',
#         conv_channels_list=None,
#         conv_kernel_size_list=None,
#         conv_dropout_rate=0.2,
#         pool_kernel_size_list=None,
#         gap=False,

#         linear_channels_list=None,
#         linear_dropout_rate=0.5,

#         seq_encoder = 'MyResNet',
#         final_layer = 'Linear',

#         final_dropout=False,
#         final_dropout_rate=0.1,
#     ):
#         super().__init__()

#         self.input_seq_length   = input_seq_length
#         self.input_seq_channels = input_seq_channels
#         self.input_feature_dim  = input_feature_dim
#         self.output_dim         = output_dim

#         self.sigmoid            = sigmoid
#         self.squeeze            = squeeze
#         self.final_dropout      = final_dropout

#         if seq_encoder == 'MyResNet':
#             self.seq_encoder = MyResNet(
#                 input_seq_length=self.input_seq_length,
#                 input_seq_channels=self.input_seq_channels,

#                 output_dim=1,
#                 sigmoid=False,
#                 squeeze=False, # squeeze should be False for concat
#                 # output_dim=self.output_dim,
#                 # sigmoid=self.sigmoid,
#                 # squeeze=self.squeeze,
                
#                 conv_first_channels=conv_first_channels,
#                 conv_first_kernel_size=conv_first_kernel_size,
#                 pool_first_kernel_size=pool_first_kernel_size,
#                 conv_padding=conv_padding,
#                 conv_activation=conv_activation,
#                 conv_layer_order=conv_layer_order,
#                 conv_channels_list=conv_channels_list,
#                 conv_kernel_size_list=conv_kernel_size_list,
#                 conv_dropout_rate=conv_dropout_rate,
#                 pool_kernel_size_list=pool_kernel_size_list,
#                 gap=gap,
#                 linear_channels_list=linear_channels_list,
#                 linear_dropout_rate=linear_dropout_rate,


#             )
#         else:
#             raise ValueError(f'seq_encoder {seq_encoder} not supported')


#         # compute the shape
#         with torch.no_grad():
#             dummy_inputs = {'seq': torch.zeros(2, self.input_seq_channels, self.input_seq_length)}
#             dummy_seq_out = self.seq_encoder(dummy_inputs)


#         self.linear_layers = nn.Sequential(OrderedDict([]))

#         for i in range(len(linear_channels_list)):
#             self.linear_layers.add_module(
#                 f'linear_{i}', nn.Linear(
#                     in_channels=dummy_seq_out.shape[-1]+input_feature_dim if i == 0 else linear_channels_list[i-1], 
#                     out_channels=linear_channels_list[i],
#                 )
#             )
#             self.linear_layers.add_module(
#                 f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
#             )
#         self.linear_layers.add_module(
#             f'linear_last', nn.Linear(
#                 in_features=current_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
#                 out_features=output_dim,
#             )
#         )


#         if final_dropout is True:
#             # self.final_dropout_layer = nn.Dropout(p=final_dropout_rate)
#             self.final_dropout_layer = ZeroMask(p=final_dropout_rate)
#         else:
#             self.final_dropout_layer = nn.Identity()


#         if final_layer == 'Linear':
#             self.final_layer = nn.Linear(dummy_seq_out.shape[1]+input_feature_dim, self.output_dim)
#         elif final_layer == 'MLP':
#             self.final_layer = MLP(
#                 input_dim=dummy_seq_out.shape[1]+input_feature_dim,
#                 output_dim=self.output_dim,
#                 hidden_dims=[100],
#                 dropout=0.5,
#             )
#         else:
#             raise ValueError(f'final_layer {final_layer} not supported')
        
#         self.sigmoid_layer = nn.Sigmoid()




#     def get_seq_out(self, inputs: dict):
#         seq = inputs.get('seq')
#         seq_out = self.seq_encoder(seq)
#         return seq_out




#     def forward(self, inputs: dict):
#         seq = inputs.get('seq')
#         feature = inputs.get('feature')

#         seq_out = self.seq_encoder(inputs)

#         if len(feature.shape) == 2:
#             # feature.shape == (batch_size, input_feature_dim)
#             feature = torch.cat([seq_out, feature], dim=-1)
#             feature = self.final_dropout_layer(feature)
#             out = self.final_layer(feature)
#         elif len(feature.shape) == 3:
#             # # feature.shape == (batch_size, input_feature_times, input_feature_dim)
#             # outs = []
#             # for i in range(feature.shape[1]):
#             #     feature_i = feature[:, i, :]
#             #     out = torch.cat([seq_out, feature_i], dim=-1)
#             #     out = self.final_layer(out)
#             #     outs.append(out)
#             # out = torch.stack(outs, dim=1)
#             # # out.shape == (batch_size, input_feature_times, output_dim)

#             # feature: (B, T, D_feat) -> (B*T, D_feat)
#             feature_reshaped = feature.view(-1, feature.size(-1))
#             seq_rep = seq_out.repeat_interleave(feature.size(1), dim=0)  # (B*T, D_seq)
            
#             feature_reshaped = torch.cat([seq_rep, feature_reshaped], dim=-1)
#             feature_reshaped = self.final_dropout_layer(feature_reshaped)
#             out = self.final_layer(feature_reshaped)
            
#             out = out.view(seq.size(0), feature.size(1), -1)              # (B, T, D_out)

#         else:
#             raise ValueError(f'feature.shape {feature.shape} not supported')

#         if self.sigmoid:
#             out = self.sigmoid_layer(out)
#         if self.squeeze:
#             out = out.squeeze(-1)
#         return out


















class MyResNetDecouple(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,
        output_dim=1,

        sigmoid=False,
        squeeze=True,

        seq_encoder_dict=None,

        zero_mask=False,
        zero_mask_rate=0.1,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.input_feature_dim  = input_feature_dim
        self.output_dim         = output_dim


        self.sigmoid            = sigmoid
        self.squeeze            = squeeze


        self.seq_encoder = utils.init_obj(models, seq_encoder_dict)

        # if seq_encoder == 'MyResNet':
        #     self.seq_encoder = MyResNet(
        #         input_seq_length=self.input_seq_length,
        #         input_seq_channels=self.input_seq_channels,

        #         output_dim=1,
        #         sigmoid=False,
        #         squeeze=False, # squeeze should be False for concat
        #         # output_dim=self.output_dim,
        #         # sigmoid=self.sigmoid,
        #         # squeeze=self.squeeze,
                
        #         conv_first_channels=conv_first_channels,
        #         conv_first_kernel_size=conv_first_kernel_size,
        #         pool_first_kernel_size=pool_first_kernel_size,
        #         conv_padding=conv_padding,
        #         conv_activation=conv_activation,
        #         conv_layer_order=conv_layer_order,
        #         conv_channels_list=conv_channels_list,
        #         conv_kernel_size_list=conv_kernel_size_list,
        #         conv_dropout_rate=conv_dropout_rate,
        #         pool_kernel_size_list=pool_kernel_size_list,
        #         gap=gap,
        #         linear_channels_list=linear_channels_list,
        #         linear_dropout_rate=linear_dropout_rate,


        # compute the shape
        with torch.no_grad():
            dummy_seq = torch.zeros(2, self.input_seq_channels, self.input_seq_length)
            dummy_feature = torch.zeros(2, self.input_feature_dim)
            dummy_seq_out = self.seq_encoder(dummy_seq)
            dummy_concat_out = torch.cat([dummy_seq_out, dummy_feature], dim=-1)
            # dummy_feature_out = self.feature_encoder(dummy_feature)
        


        self.zero_mask_layer = ZeroMask(p=zero_mask_rate) if zero_mask else nn.Identity()


        self.linear_layers = nn.Sequential(OrderedDict([]))

        current_channel = dummy_concat_out.shape[1]
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_{i}', nn.Linear(
                    in_features=current_channel, 
                    out_features=linear_channels_list[i],
                )
            )
            self.linear_layers.add_module(
                f'linear_activation_{i}', nn.ReLU()
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
            current_channel = linear_channels_list[i]

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=current_channel, 
                out_features=output_dim,
            )
        )
        
        self.sigmoid_layer = nn.Sigmoid()




    def get_seq_out(self, inputs: dict):
        seq = inputs.get('seq')
        seq_out = self.seq_encoder(seq)
        return seq_out




    def forward(self, inputs: dict):
        seq = inputs.get('seq')
        feature = inputs.get('feature')

        seq_out = self.seq_encoder(inputs)

        if len(feature.shape) == 2:
            # feature.shape == (batch_size, input_feature_dim)
            concat_out = torch.cat([seq_out, feature], dim=-1)
            mask_out = self.zero_mask_layer(concat_out)
            out = self.linear_layers(mask_out)
        elif len(feature.shape) == 3:
            # # feature.shape == (batch_size, input_feature_times, input_feature_dim)
            # outs = []
            # for i in range(feature.shape[1]):
            #     feature_i = feature[:, i, :]
            #     out = torch.cat([seq_out, feature_i], dim=-1)
            #     out = self.final_layer(out)
            #     outs.append(out)
            # out = torch.stack(outs, dim=1)
            # # out.shape == (batch_size, input_feature_times, output_dim)

            # feature: (B, T, D_feat) -> (B*T, D_feat)
            feature_reshaped = feature.view(-1, feature.size(-1))
            seq_rep = seq_out.repeat_interleave(feature.size(1), dim=0)  # (B*T, D_seq)
            
            feature_reshaped = torch.cat([seq_rep, feature_reshaped], dim=-1)
            feature_reshaped = self.zero_mask_layer(feature_reshaped)
            out = self.linear_layers(feature_reshaped)
            
            out = out.view(seq.size(0), feature.size(1), -1)              # (B, T, D_out)

        else:
            raise ValueError(f'feature.shape {feature.shape} not supported')

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out






















if __name__ == '__main__':

    yaml_str = '''
model:
    type: MyResNet
    args:
        input_seq_length:       200
        input_seq_channels:     4
        output_dim:             1
        sigmoid:                False
        squeeze:                True

        conv_channels_list: [256, 256, 256, 256, 256, 256]
        conv_kernel_size_list: [3, 3, 3, 3, 3, 3]
        conv_padding_list: [1, 1, 1, 1, 1, 1]
        pool_kernel_size_list: [2,2,2,2,2,2]
        conv_dropout_rate: 0.2
        gap: true

        linear_channels_list: [1024]
        linear_dropout_rate: 0.5
        '''
    import yaml
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = utils.init_obj(models, config['model'])

    seq = torch.zeros(size=(1, 4, 200))
    inputs = {'seq': seq}
    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )
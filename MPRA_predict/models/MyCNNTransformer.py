import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import einops
from rotary_embedding_torch import RotaryEmbedding


class ConvBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            gn_group_size = 16,):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # 是gelu不是relu，非常重要！?
        self.gelu = nn.GELU()
        if gn_group_size is None:
            self.gn = nn.BatchNorm1d(out_channels)
        else:
            assert out_channels % gn_group_size == 0
            gn_num_groups = out_channels // gn_group_size
            self.gn = nn.GroupNorm(gn_num_groups, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.gelu(out)
        out = self.gn(out)
        return out


class LinearBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, d_embed, n_heads, dropout_rate=0.1, use_position_embedding=True):
        super().__init__()
        assert d_embed % n_heads == 0
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)

        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.out_linear = nn.Linear(d_embed, d_embed)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask=None):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = batch_size, seq_len, self.n_heads, self.d_head
        q = self.q_linear(x).view(interim_shape).transpose(1, 2)
        k = self.k_linear(x).view(interim_shape).transpose(1, 2)
        v = self.v_linear(x).view(interim_shape).transpose(1, 2)

        if self.use_position_embedding:
            # q and k have shape (batch_size, n_heads, seq_len, d_head)
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        weight = einops.einsum(q, k, 'b h q d, b h k d -> b h q k')
        #weight = torch.matmul(q, k.transpose(-2, -1))
        weight = weight / np.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        output = einops.einsum(weight, v, 'b h q k, b h k d -> b h q d')
        #output = torch.matmul(weight, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        output = self.out_linear(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_mlp, dropout_rate=0.1, bias=False, use_position_embedding=True):
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        self.attn = SelfAttention(
            d_embed, 
            n_heads, 
            dropout_rate, 
            use_position_embedding)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_embed),
        )

        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, input):

        input = self.layer_norm1(input)
        attn_output = self.attn(input)
        attn_output = self.dropout1(attn_output)
        mlp_input = attn_output + input

        mlp_input = self.layer_norm2(mlp_input)
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.dropout2(mlp_output)
        output = mlp_output + mlp_input

        return output












class MyCNNTransformer(nn.Module):
    def __init__(
            self, 
            input_seq_length=200,
            input_feature_shape=(4),
            output_dim=1,

            sigmoid=False,
            squeeze=True,

            conv_channels_list=None,
            conv_kernel_size_list=None,
            conv_padding_list=None,
            pool_kernel_size_list=None,
            pool_padding_list=None,
            conv_dropout_rate=0.2,
            global_average_pooling=False,

            num_trans_blocks=3, 
            trans_d_embed=256, 
            trans_n_heads=8, 
            trans_d_mlp=256,
            trans_dropout_rate=0.1,
            trans_output='cls',

            linear_channels_list=None,
            linear_dropout_rate=0.5,
        ):
        super().__init__()

        self.input_seq_length       = input_seq_length
        self.input_feature_shape    = input_feature_shape
        self.output_dim             = output_dim
        self.sigmoid                = sigmoid
        self.squeeze                = squeeze
        self.trans_output           = trans_output

        if conv_padding_list is None:
            conv_padding_list = [0] * len(conv_kernel_size_list)
        if pool_padding_list is None:
            pool_padding_list = [0] * len(pool_kernel_size_list)

        self.conv_layers = nn.Sequential(OrderedDict([]))
        for i in range(len(conv_kernel_size_list)):
            self.conv_layers.add_module(
                f'conv_block_{i}', ConvBlock(
                    in_channels=4 if i == 0 else conv_channels_list[i-1], 
                    out_channels=conv_channels_list[i], 
                    kernel_size=conv_kernel_size_list[i], 
                    stride=1, 
                    padding=conv_padding_list[i]))
                
            self.conv_layers.add_module(
                f'max_pool_{i}', nn.MaxPool1d(
                    kernel_size=pool_kernel_size_list[i], 
                    padding=pool_padding_list[i],
                    ceil_mode = True))

            self.conv_layers.add_module(
                f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate))

        if global_average_pooling:
            self.conv_layers.add_module(
                f'gap_layer', nn.AdaptiveAvgPool1d(1))

        # if input_feature_shape == None:
        #     self.cls_embedding_layer = nn.Embedding(1, trans_d_embed)
        # elif len(input_feature_shape) == 1:
        #     self.cls_embedding_layer = nn.Embedding(input_feature_shape[0], trans_d_embed)
        # elif len(input_feature_shape) == 2:
        #     self.cls_embedding_layer = nn.Embedding(input_feature_shape[1], trans_d_embed)
        # else:
        #     raise ValueError('input_feature_shape must be 0, 1 or 2')
        # nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02) # 初始化可能很重要！

        if input_feature_shape == None:
            self.cls_embedding_layer = nn.Linear(1, trans_d_embed)
        elif len(input_feature_shape) == 1:
            self.cls_embedding_layer = nn.Linear(input_feature_shape[0], trans_d_embed)
        elif len(input_feature_shape) == 2:
            self.cls_embedding_layer = nn.Linear(input_feature_shape[1], trans_d_embed)
        else:
            raise ValueError('input_feature_shape must be 0, 1 or 2')
        # nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02) # 初始化可能很重要！


        self.trans_layers = nn.Sequential(OrderedDict([]))
        for i in range(num_trans_blocks):
            self.trans_layers.add_module(
                f'transformer_block_{i}', TransformerBlock(
                    d_embed=trans_d_embed, 
                    n_heads=trans_n_heads, 
                    d_mlp=trans_d_mlp, 
                    dropout_rate=trans_dropout_rate))
        
        # with torch.no_grad():
        #     test_input = torch.zeros(1, 4, self.input_seq_length)
        #     test_output = self.conv_layers(test_input)
        #     hidden_dim = test_output[0].reshape(-1).shape[0]

        self.linear_layers = nn.Sequential(OrderedDict([]))
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=trans_d_embed if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i]))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate))

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=trans_d_embed if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                out_features=output_dim))

        self.sigmoid_layer = nn.Sigmoid()



    def forward_with_feature(self, seq, feature):
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)

        seq = self.conv_layers(seq)
        seq = seq.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)

        cls = self.cls_embedding_layer(feature)
        cls = cls.unsqueeze(1)
        # print(seq.shape, cls.shape)

        seq = torch.concat([cls, seq], dim=1)
        seq = self.trans_layers(seq)

        if self.trans_output == 'cls':
            cls = seq[:, 0]
        elif self.trans_output == 'seq':
            cls = seq[:, 1:].mean(1)
        cls = cls.view(cls.size(0), -1)
        out = self.linear_layers(cls)

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out



    def forward(self, inputs):
        
        if self.input_feature_shape == None:
            if isinstance(inputs, dict):
                seq = inputs.get('seq')
            elif isinstance(inputs, (list, tuple)):
                seq = inputs[0]
            else:
                raise ValueError('inputs type must be dict or list or tuple or tensor')

            pseudo_feature = torch.zeros(seq.size(0), 1).to(seq.device)
            out = self.forward_with_feature(seq, pseudo_feature)
            return out



        elif len(self.input_feature_shape) == 1:
            if isinstance(inputs, dict):
                seq = inputs.get('seq')
                feature = inputs.get('feature')
            elif isinstance(inputs, (list, tuple)):
                seq = inputs[0]
                feature = inputs[1]
            else:
                raise ValueError('inputs type must be dict or list or tuple or tensor')
            
            out = self.forward_with_feature(seq, feature)
            return out



        elif len(self.input_feature_shape) == 2:
            if isinstance(inputs, dict):
                seq = inputs.get('seq')
                feature = inputs.get('feature')
            elif isinstance(inputs, (list, tuple)):
                seq = inputs[0]
                feature = inputs[1]
            else:
                raise ValueError('inputs type must be dict or list or tuple or tensor')
            
            outs = []
            for i in range(self.input_feature_shape[0]):
                feature_i = feature[:, i, :]  # cell type i features
                out = self.forward_with_feature(seq, feature_i)
                outs.append(out)
            outs = torch.stack(outs, dim=1)  # (batch_size, num_cell_types)
            return outs







if __name__ == '__main__':
    import yaml
    import torchinfo
    from .. import models, utils










    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: null
                output_dim: 5
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    inputs = {'seq': seq}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)



















    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: [4]
                output_dim: 5
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    feature = torch.zeros(2, 4)
    inputs = {'seq': seq, 'feature': feature}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)





    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: [5,4]
                output_dim: 1
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    feature = torch.zeros(2, 5, 4)
    inputs = {'seq': seq, 'feature': feature}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)

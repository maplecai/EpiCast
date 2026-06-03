import torch
import torch.nn as nn

from .conv_block import ConvBlock, ResConvBlock
from .linear_block import LinearBlock
from .trans_block import TransBlock
from .film import FiLM

class ConvTransformerFeature(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,

        output_dim=1,
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

        num_linear_blocks=1,
        linear_channels=1024,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.input_feature_dim  = input_feature_dim
        self.output_dim         = output_dim
        self.last_activation    = last_activation
        self.squeeze            = squeeze
        self.fusion_type        = fusion_type

        self.trans_output_mode  = trans_output_mode
        self.trans_add_cls      = trans_add_cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))
        self.cls_token_len = int(self.trans_add_cls)
        self.seq_token_len = 0
        self.epi_token_len = 0

        self.conv_layers = nn.Sequential()
        for i in range(num_conv_blocks):
            self.conv_layers.add_module(
                f'conv_block_{i}', ResConvBlock(
                    in_channels=input_seq_channels if i == 0 else conv_channels, 
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
        
        if self.fusion_type == 'film':
            self.fusion_layer = FiLM(
                num_features=trans_d_embed,
                cond_dim=input_feature_dim,
                hidden_dim=0,
                channel_dim=2,
                affine=True,
                init_identity=True,
            )
        elif self.fusion_type == 'concat':
            self.feature_embedding_layer = nn.Linear(input_feature_dim, trans_d_embed)
            self.token_type_embedding_layer = nn.Embedding(3, trans_d_embed)

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


    def forward_conv(self, seq: torch.Tensor):
        seq = seq.permute(0, 2, 1)
        seq_emb = self.conv_layers(seq)
        seq_emb = seq_emb.permute(0, 2, 1)
        return seq_emb

    def forward_trans(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, H = tokens.shape
        if self.trans_add_cls:
            tokens = torch.cat([self.cls_token.expand(B, 1, -1), tokens], 1)

        out = self.trans_layers(tokens)

        if self.trans_output_mode == 'cls':
            out = out[:, 0]
        elif self.trans_output_mode == 'seq_avg':
            start = self.cls_token_len
            end = out.shape[1] - self.epi_token_len
            out = out[:, start:end].mean(1)
        elif self.trans_output_mode == 'seq_all':
            start = self.cls_token_len
            end = out.shape[1] - self.epi_token_len
            out = out[:, start:end]
        elif self.trans_output_mode == 'seq_flatten':
            start = self.cls_token_len
            end = out.shape[1] - self.epi_token_len
            out = out[:, start:end].reshape(B, -1)
        elif self.trans_output_mode == 'all':
            out = out
        else:
            raise ValueError(f"Unsupported trans_output_mode mode: {self.trans_output_mode}")

        return out

    def forward_fusion(self, seq_emb: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == 'film':
            emb = self.fusion_layer(seq_emb, feature)
            self.seq_token_len = seq_emb.shape[1]
            self.epi_token_len = 0
            return emb
        
        elif self.fusion_type == 'concat':
            batch_size = seq_emb.shape[0]
            device = seq_emb.device
            feature_emb = self.feature_embedding_layer(feature).unsqueeze(1)
            # print(seq_emb.shape, feature_emb.shape)
            tokens = torch.cat([seq_emb, feature_emb], dim=1)

            self.seq_token_len = seq_emb.shape[1]
            self.epi_token_len = feature_emb.shape[1]

            token_type_ids = torch.cat([
                torch.full((batch_size, self.cls_token_len), 0, dtype=torch.long, device=device), # CLS token
                torch.full((batch_size, self.seq_token_len), 1, dtype=torch.long, device=device), # Seq tokens
                torch.full((batch_size, self.epi_token_len), 2, dtype=torch.long, device=device), # Epi tokens
            ], dim=1)
            token_type_embed = self.token_type_embedding_layer(token_type_ids)
            tokens = tokens + token_type_embed
            return tokens


    def forward_linear(self, emb: torch.Tensor):
        out = self.linear_layers(emb)
        if self.squeeze:
            out = out.squeeze(-1)
        return out

    def forward(self, inputs: dict | list) -> torch.Tensor:
        if isinstance(inputs, dict):
            seq = inputs.get('seq', None)
            feature = inputs.get('feature', None)
        elif isinstance(inputs, list):
            seq = inputs[0]
            feature = inputs[1]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        batch_size = seq.shape[0]
        expected_shape = (batch_size, self.input_seq_length, self.input_seq_channels)
        assert seq.shape == expected_shape, f"{seq.shape = }, {expected_shape = }"

        seq_emb = self.forward_conv(seq)

        if len(feature.shape) == 2:
            emb = self.forward_fusion(seq_emb, feature)
            emb = self.forward_trans(emb)
            out = self.forward_linear(emb)
        
        elif len(feature.shape) == 3:
            features = feature
            B, C, D = features.shape
            outs = []
            for i in range(C):  # per cell type
                emb = self.forward_fusion(seq_emb, features[:, i, :])
                emb = self.forward_trans(emb)
                out = self.forward_linear(emb)
                outs.append(out)
            out = torch.stack(outs, dim=1)  # (batch_size, num_cell_types)
            if self.squeeze:
                out = out.squeeze(-1)

        else:
            raise ValueError(f'Inval {feature.shape=}')
        
        return out



if __name__ == '__main__':
    import torchinfo

    model = ConvTransformerFeature(
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=4,
        trans_output_mode = 'seq_avg',
        
    )

    seq = torch.zeros(size=(2, 200, 4))
    feature = torch.zeros(size=(2, 5, 4))
    inputs = {'seq': seq, 'feature': feature}

    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )
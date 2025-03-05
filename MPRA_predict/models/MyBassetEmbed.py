# from .MyBasset import *
# from .. import models
# from .. import utils
# import yaml


# class MyBassetMultiTask(nn.Module):
#     def __init__(
#         self, 
#         input_length = 1000,
#         output_dim = 2,
#         integration_strategy = 'concat',
#         **kwargs,
#     ):
#         super().__init__()

#         self.integration_strategy = integration_strategy

#         self.task_encoder = TaskEncoder(**kwargs.get('task_encoder', {}))
#         self.seq_encoder = ConvLayers(**kwargs.get('seq_encoder', {}))

#         with torch.no_grad():
#             test_input = torch.randn(1, 4, input_length)
#             test_output = self.seq_encoder(test_input)
#             hidden_dim = test_output.view(test_output.size(0), -1).shape[1]

#         self.decoder = LinearLayers(input_dim=hidden_dim, output_dim=output_dim, **kwargs.get('decoder', {}))

#         if integration_strategy == 'concat':
#             embedding_dim = self.task_encoder.embedding_dim
#             self.transform_layer = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

#         elif integration_strategy == 'cross_attention':
#             self.cross_attention = CrossAttention(**kwargs.get('cross_attention', {}))
#             self.transform_layer = nn.Linear(self.cross_attention.d_embed, hidden_dim)
        
#         elif integration_strategy == 'add':
#             self.transform_layer = nn.Linear(self.task_encoder.embedding_dim, hidden_dim)

#         elif integration_strategy == 'none':
#             pass


#     def forward(self, x, cell_idx):

#         x = self.seq_encoder(x)
#         # x.shape = (batch_size, length, n_channels)
        
#         task_embedding = self.task_encoder(cell_idx)
#         # task_embedding.shape = (batch_size, added_dim)

#         if self.integration_strategy == 'concat':
#             x = x.view(x.size(0), -1)
#             task_embedding = task_embedding.view(task_embedding.size(0), -1)
#             x = torch.cat([x, task_embedding], dim=1)
#             x = self.transform_layer(x)

#         elif self.integration_strategy == 'cross_attention':
#             x = x.permute(0, 2, 1)
#             # x.shape = (batch_size, length, n_channels)
#             task_embedding = task_embedding.unsqueeze(1).float()
#             # task_embedding.shape = (batch_size, 1, added_dim)
#             x = self.cross_attention(task_embedding, x)
#             x = x.view(x.size(0), -1)
#             x = self.transform_layer(x)
#             # x.shape = (batch_size, 1, n_channels)

#         elif self.integration_strategy == 'add':
#             # task_embedding.shape = (batch_size, added_dim)
#             task_embedding = self.transform_layer(task_embedding)
#             # task_embedding.shape = (batch_size, hidden_dim)
#             x = x.view(x.size(0), -1)
#             x = x + task_embedding
        
#         elif self.integration_strategy == 'none':
#             pass

#         x = x.view(x.size(0), -1)
#         x = self.decoder(x)

#         return x





# class MyBassetEmbed(nn.Module):
#     def __init__(
#         self, 
#         input_length=200,
#         output_dim=1,
#         squeeze=True,
#         integration_strategy = 'concat',
#         embedding_dim = 1,
#         **kwargs,
#     ):
#         super().__init__()

#         self.input_length = input_length
#         self.output_dim = output_dim
#         self.squeeze = squeeze
#         self.integration_strategy = integration_strategy
#         self.embedding_dim = embedding_dim

#         self.encoder = utils.init_obj(models, kwargs.get('encoder_kwarg', {}))

#         with torch.no_grad():
#             x = torch.randn(2, 4, input_length)
#             x = self.encoder(x)
#             x = x.view(x.size(0), -1)
#             hidden_dim = x.shape[1]
        
#         if integration_strategy == 'concat':
#             self.transform_layer = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        
#         # elif integration_strategy == 'add':
#         #     self.transform_layer = nn.Linear(embedding_dim, hidden_dim)

#         # elif integration_strategy == 'cross_attention':
#         #     self.cross_attention = CrossAttention(**kwargs.get('cross_attention', {}))
#         #     self.transform_layer = nn.Linear(self.cross_attention.d_embed, hidden_dim)

#     def forward(self, x, embedding):
#         if x.shape[2] == 4:
#             x = x.transpose(1, 2)

#         x = self.encoder(x)

#         if self.integration_strategy == 'concat':
#             x = x.view(x.size(0), -1)
#             x = torch.cat([x, embedding], dim=1)
#             x = self.transform_layer(x)

#         # elif self.integration_strategy == 'add':
#         #     x = x.view(x.size(0), -1)
#         #     # embedding.shape = (batch_size, added_dim)
#         #     embedding = self.transform_layer(embedding)
#         #     # embedding.shape = (batch_size, hidden_dim)
#         #     x = x + embedding

#         # elif self.integration_strategy == 'cross_attention':
#         #     x = x.permute(0, 2, 1)
#         #     # x.shape = (batch_size, length, n_channels)
#         #     task_embedding = task_embedding.unsqueeze(1).float()
#         #     # task_embedding.shape = (batch_size, 1, added_dim)
#         #     x = self.cross_attention(task_embedding, x)
#         #     x = x.view(x.size(0), -1)
#         #     x = self.transform_layer(x)
#         #     # x.shape = (batch_size, 1, n_channels)

#         if self.squeeze:
#             x = x.squeeze(-1)

#         return x



# if __name__ == '__main__':

#     yaml_str = '''
# model:
#     type: MyBassetSequential
#     args:
#         input_length:   200
#         output_dim: 1
#         n_cells:    2
#         n_tasks:    2

#         encoder_kwargs:
#         -   type: MyBassetEncoder
#             args:
#                 conv_channels_list:     [256,256,256,256]
#                 conv_kernel_size_list:  [7,7,7,7]
#                 conv_padding_list:      [0,0,0,0]
#                 pool_kernel_size_list:  [2,2,2,2]
#                 conv_dropout_rate:      0.2
                
#         shared_decoder_kwargs:
#         -   type: MyBassetDecoder
#             args:
#                 linear_channels_list:   [256,256]
#                 linear_dropout_rate:    0.5
#                 last_linear_layer:      false
#                 sigmoid:                false

#         cell_decoder_kwargs:
#         -   type: MyBassetDecoder
#             args:
#                 input_dim:              256
#                 linear_channels_list:   [256,256]
#                 linear_dropout_rate:    0.5
#                 last_linear_layer:      false
#                 sigmoid:                false
#         -   type: MyBassetDecoder
#             args:
#                 input_dim:              256
#                 linear_channels_list:   [256,256]
#                 linear_dropout_rate:    0.5
#                 last_linear_layer:      false
#                 sigmoid:                false
                
#         task_decoder_kwargs:
#         -   type: MyBassetDecoder
#             args:
#                 input_dim:              256
#                 output_dim:             1
#                 linear_channels_list:   [256,256]
#                 linear_dropout_rate:    0.5
#                 last_linear_layer:      true
#                 sigmoid:                true
#         -   type: MyBassetDecoder
#             args:
#                 input_dim:              256
#                 output_dim:             1
#                 linear_channels_list:   [256,256]
#                 linear_dropout_rate:    0.5
#                 last_linear_layer:      true
#                 sigmoid:                false
#     '''

#     config = yaml.load(yaml_str, Loader=yaml.FullLoader)
#     model = utils.init_obj(models, config['model'])

#     x = torch.randn(2, 200, 4)
#     t = torch.tensor([0, 1])
#     output = model(x, t, t)
#     torchinfo.summary(model, input_data=(x, t, t), depth=4)

# from .MyBasset import *
# from .Attention import *
# import yaml
# from ..utils import *
# from . import MyBasset
# from .. import models

# class OnehotEncoder(nn.Module):
#     def __init__(self, num_classes, onehot_table=None):
#         super().__init__()

#         if onehot_table is None:
#             self.num_classes = num_classes
#             self.onehot_table = None
#         else:
#             self.onehot_table = nn.Parameter(torch.tensor(onehot_table, dtype=torch.float32), requires_grad=False)
#             self.num_classes = self.onehot_table[0].shape[-1]

#     def forward(self, x):
#         if self.onehot_table is None:
#             return F.one_hot(x, num_classes=self.num_classes)
#         else:
#             return self.onehot_table[x.long()]
#         # return F.one_hot(x, num_classes=self.num_classes)


# # class TableEncoder(nn.Module):
# #     def __init__(self, weight: torch.Tensor=None):
# #         super().__init__()
# #         self.weight = nn.Parameter(weight, requires_grad=False)

# #     def forward(self, x):
# #         task_encoding = self.weight[x.long()]
# #         return task_encoding



# # class TaskEncoderMultiInput(nn.Module):
# #     def __init__(self, num_tasks_list):
# #         super().__init__()
# #         self.num_tasks_list = num_tasks_list

# #     def forward(self, x):
# #         # x.shape = (batch_size, n)
# #         task_encoding = []
# #         for i in range(len(x[0])):
# #             task_encoding.append(F.one_hot(x[:,i], num_classes=self.num_tasks_list[i]))
# #         task_encoding = torch.cat(task_encoding, dim=1)
# #         return task_encoding




# class TaskEncoder(nn.Module):
#     def __init__(self, encoding_mode='embedding', num_tasks=4, embedding_dim=16, onehot_table=None):
#         super().__init__()
#         self.encoding_mode = encoding_mode
#         self.num_tasks = num_tasks

#         self.onehot_layer = OnehotEncoder(num_tasks, onehot_table=onehot_table)

#         if encoding_mode == 'embedding':
#             # 训练
#             #self.embedding_layer = nn.Embedding(self.onehot_layer.num_classes, embedding_dim)
#             self.embedding_layer = nn.Linear(self.onehot_layer.num_classes, embedding_dim)
#             self.embedding_dim = embedding_dim
#         elif encoding_mode == 'identity':
#             # 不训练
#             self.embedding_layer = nn.Identity()
#             self.embedding_dim = self.onehot_layer.num_classes

#     def forward(self, x):
#         x = self.onehot_layer(x)
#         x = x.float()
#         x = self.embedding_layer(x)
#         return x



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



# class MyBassetMultiCellOutput(nn.Module):
#     def __init__(
#         self, 
#         sigmoid_list = None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.my_basset_multi_task = MyBassetMultiTask(**kwargs)
#         self.sigmoid_list = sigmoid_list
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, cell_idx, output_idx):
#         # 假设一个batch内只有一种类型的数据
#         output_idx = output_idx[0].long()

#         x = self.my_basset_multi_task(x, cell_idx)
#         # x.shape = (batch_size, n_output)
#         x = x[:, output_idx]

#         if self.sigmoid_list is not None:
#             if self.sigmoid_list[output_idx] == True:
#                 x = self.sigmoid(x)
        
#         return x



# if __name__ == '__main__':

#     # model_kwargs = {
#     #     'input_length': 1000,
#     #     'output_dim': 1,
#     #     'task_encoder': {
#     #         'encoding_mode': 'embedding',
#     #         'num_tasks': 2,
#     #         'embedding_dim': 16,
#     #         'onehot_table': [[1,0,1,0],[0,1,0,1]],
#     #     },
#     #     'integration_strategy': 'cross_attention',
#     #     'cross_attention': {
#     #         'n_heads': 1,
#     #         'd_embed': 16,
#     #         'd_cross': 100,
#     #     },
#     #     'seq_encoder': {
#     #         'conv_channels_list': [100]*12,
#     #         'conv_kernel_size_list': [3]*12,
#     #         'pool_kernel_size_list': [1,1,1,4]*3,
#     #     },
#     #     'decoder': {
#     #         'linear_channels_list': [100,100],
#     #     }
#     # }

#     # model = MyBassetMultiTask(**model_kwargs)
#     # x = torch.randn(2, 1000, 4)
#     # t = torch.tensor([0, 1])
#     # output = model(t, x)
#     # torchinfo.summary(model, input_data=(t, x))



#     # model_kwargs = {
#     #     'input_length': 1000,
#     #     'output_dim': 1,
#     #     'task_encoder': {
#     #         'encoding_mode': 'embedding',
#     #         'num_tasks': 2,
#     #         'embedding_dim': 16,
#     #         'onehot_table': [[1,0,1,0],[0,1,0,1]],
#     #     },
#     #     'integration_strategy': 'concat',
#     #     'seq_encoder': {
#     #         'conv_channels_list': [100]*12,
#     #         'conv_kernel_size_list': [3]*12,
#     #         'pool_kernel_size_list': [1,1,1,4]*3,
#     #     },
#     #     'decoder': {
#     #         'linear_channels_list': [100,100],
#     #     }
#     # }

#     # model = MyBassetMultiTask(**model_kwargs)
#     # x = torch.randn(2, 1000, 4)
#     # t = torch.tensor([0, 1])
#     # output = model(t, x)
#     # print(output.shape)
#     # torchinfo.summary(model, input_data=(t, x))



#     model_kwargs = {
#         'input_length': 1000,
#         'output_dim': 1,
#         'task_encoder': {
#             'encoding_mode': 'embedding',
#             'num_tasks': 2,
#             'embedding_dim': 16,
#             'onehot_table': [[1,0,1,0],[0,1,0,1]],
#         },
#         'integration_strategy': 'add',
#         'seq_encoder': {
#             'conv_channels_list': [100]*12,
#             'conv_kernel_size_list': [3]*12,
#             'pool_kernel_size_list': [1,1,1,4]*3,
#         },
#         'decoder': {
#             'linear_channels_list': [100,100],
#         }
#     }

#     model = MyBassetMultiTask(**model_kwargs)
#     x = torch.randn(2, 1000, 4)
#     t = torch.tensor([0, 1])
#     output = model(t, x)
#     print(output.shape)
#     torchinfo.summary(model, input_data=(t, x))


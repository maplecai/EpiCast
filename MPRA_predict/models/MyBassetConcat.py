# from .MyBasset import *
# from .. import models
# from .. import utils


# class MyBassetConcat(nn.Module):
#     def __init__(
#         self, 
#         input_length=200,
#         output_dim=1,
#         squeeze=True,
#         n_cells=2,
#         n_tasks=2,
#         **kwargs,
#     ):
#         super().__init__()

#         self.input_length = input_length
#         self.output_dim = output_dim
#         self.squeeze = squeeze
#         self.n_cells = n_cells
#         self.n_tasks = n_tasks

#         self.encoder = utils.init_obj(models, kwargs.get('encoder_kwarg', {}))

#         with torch.no_grad():
#             x = torch.randn(2, 4, input_length)
#             x = self.encoder(x)
#             x = x.view(x.size(0), -1)
#             hidden_dim = x.shape[1]

#         # self.processor = utils.init_obj(models, kwargs.get('processor_kwargs', {}), input_dim=hidden_dim)

#         # with torch.no_grad():
#         #     x = self.processor(x)
#         #     x = x.view(x.size(0), -1)
#         #     hidden_dim = x.shape[1]
        
#         self.cell_decoder_list = nn.ModuleList([
#             utils.init_obj(models, kwargs.get('decoder_kwarg', {}), input_dim=hidden_dim) for i in range(n_cells)])
        
#         self.task_decoder_list = nn.ModuleList([
#             utils.init_obj(models, kwargs.get('decoder_kwarg', {}), input_dim=hidden_dim) for i in range(n_tasks)])

#         with torch.no_grad():
#             x = self.cell_decoder_list[0](x)
#             hidden_dim = x.shape[1]
        
#         self.last_decoder = utils.init_obj(models, kwargs.get('last_decoder_kwarg', {}), input_dim=2*hidden_dim, output_dim=output_dim)


#     def forward(self, x, cell_idx, task_idx):
#         if x.size(1) != 4 and x.size(2) == 4:
#             x = x.transpose(1, 2)
#         # 假设一个batch内只有一种task cell的数据
#         # print(cell_idx.shape, output_idx.shape)
#         if cell_idx.dim() > 0:
#             cell_idx = cell_idx[0].long()
#         if task_idx.dim() > 0:
#             task_idx = task_idx[0].long()

#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         x1 = self.cell_decoder_list[cell_idx](x)
#         x2 = self.task_decoder_list[task_idx](x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.last_decoder(x)

#         if self.squeeze:
#             x = x.squeeze(-1)

#         return x



# class MyBassetSequential(nn.Module):
#     def __init__(
#         self, 
#         input_length=200,
#         output_dim=1,
#         n_cells=2,
#         n_tasks=2,
#         squeeze=True,
#         **kwargs,
#     ):
#         super().__init__()

#         self.input_length = input_length
#         self.output_dim = output_dim
#         self.n_cells = n_cells
#         self.n_tasks = n_tasks
#         self.squeeze = squeeze

#         self.encoder = utils.init_obj(models, kwargs['encoder_kwargs'][0])

#         with torch.no_grad():
#             x = torch.randn(2, 4, input_length)
#             x = self.encoder(x)
#             x = x.view(x.size(0), -1)
#             hidden_dim = x.shape[1]

#         self.shared_decoder = utils.init_obj(models, kwargs['shared_decoder_kwargs'][0], input_dim=hidden_dim)

#         self.cell_decoder_list = nn.ModuleList([
#             utils.init_obj(models, kwargs['cell_decoder_kwargs'][i]) for i in range(n_cells)])
        
#         self.task_decoder_list = nn.ModuleList([
#             utils.init_obj(models, kwargs['task_decoder_kwargs'][i]) for i in range(n_tasks)])



#     def forward(self, x, cell_idx, task_idx):
#         if x.shape[2] == 4:
#             x = x.transpose(1, 2)
#         # 假设一个batch内只有一种task cell的数据
#         # print(cell_idx.shape, output_idx.shape)
#         if cell_idx.dim() > 0:
#             cell_idx = cell_idx[0].long()
#         if task_idx.dim() > 0:
#             task_idx = task_idx[0].long()

#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         x = self.shared_decoder(x)
#         x = self.cell_decoder_list[cell_idx](x)
#         x = self.task_decoder_list[task_idx](x)

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

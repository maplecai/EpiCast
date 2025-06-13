# import math
# import torch
# from torch import nn
# from torch.nn import functional as F
# # from torch.nn import TransformerEncoderLayer

# class SelfAttention(nn.Module):
#     def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
#         super().__init__()
#         self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
#         self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
#         self.n_heads = n_heads
#         self.d_head = d_embed // n_heads

#     def forward(self, x, causal_mask=False):
#         input_shape = x.shape
#         batch_size, sequence_length, d_embed = input_shape
#         interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

#         q, k, v = self.in_proj(x).chunk(3, dim=-1)

#         q = q.view(interim_shape).transpose(1, 2)
#         k = k.view(interim_shape).transpose(1, 2)
#         v = v.view(interim_shape).transpose(1, 2)

#         weight = q @ k.transpose(-1, -2)
#         if causal_mask:
#             mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
#             weight.masked_fill_(mask, -torch.inf)
#         weight /= math.sqrt(self.d_head)
#         weight = F.softmax(weight, dim=-1)

#         output = weight @ v
#         output = output.transpose(1, 2)
#         output = output.reshape(input_shape)
#         output = self.out_proj(output)
#         return output


# class CrossAttention(nn.Module):
#     def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
#         """
#         初始化多头注意力机制的参数。
        
#         参数:
#         - n_heads (int): 注意力头的数量。
#         - d_embed (int): 嵌入维度的大小。
#         - d_cross (int): 交叉维度的大小，通常用于编码-解码注意力中。
#         - in_proj_bias (bool): 是否在输入投影中使用偏置，默认为True。
#         - out_proj_bias (bool): 是否在输出投影中使用偏置，默认为True。
#         """
#         super().__init__()  # 初始化父类
#         self.n_heads = n_heads  # 注意力头的数量
#         self.d_embed = d_embed  # 嵌入维度的大小
#         self.d_cross = d_cross  # 交叉维度的大小
        
#         # 初始化查询、键、值的线性变换投影
#         self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)  # 查询投影
#         self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # 键投影
#         self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # 值投影
        
#         # 初始化输出投影的线性变换
#         self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
#         # 确保嵌入维度可以被注意力头数整除
#         assert d_embed % n_heads == 0
        
#         # 计算每个注意力头的维度
#         self.d_head = d_embed // n_heads
    
#     def forward(self, x, y):
#         # print(x.shape, y.shape)
#         # print(self.d_embed, self.d_cross)

#         input_shape = x.shape
#         batch_size, sequence_length, d_embed = input_shape
#         interim_shape = (batch_size, -1, self.n_heads, self.d_head)

#         q = self.q_proj(x)
#         k = self.k_proj(y)
#         v = self.v_proj(y)

#         q = q.view(interim_shape).transpose(1, 2)
#         k = k.view(interim_shape).transpose(1, 2)
#         v = v.view(interim_shape).transpose(1, 2)

#         weight = q @ k.transpose(-1, -2)
#         weight /= math.sqrt(self.d_head)
#         weight = F.softmax(weight, dim=-1)

#         output = weight @ v
#         output = output.transpose(1, 2).contiguous()
#         output = output.view(input_shape)
#         output = self.out_proj(output)
#         return output


# if __name__ == '__main__':

#     cross_attention = CrossAttention(n_heads=1, d_embed=16, d_cross=13)

#     query = torch.rand((10, 100, 13))
#     key = torch.rand((10, 1, 16))

#     output = cross_attention(key, query)

#     print(output.shape)


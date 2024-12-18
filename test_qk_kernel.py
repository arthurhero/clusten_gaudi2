#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
import habana_frameworks.torch.core as htcore
from clusten import CLUSTENQKFunction

"""
Test the correctness of QK custom kernel
"""

b = 2 
h = 2
n = 196
m = 4
n_ = n*m
c = 32

device = torch.device("hpu")

# dummy data
query = nn.Parameter(torch.randn(b, h, n, c)).to(device)
query.retain_grad()
key = nn.Parameter(torch.randn(b, h, n_, c)).to(device)
key.retain_grad()
#nn_idx = torch.randint(n, (b, n, m)).to(device)
nn_idx = torch.arange(n_).reshape(1,n,m).expand(b,-1,-1).contiguous().to(device) 

# use the custom kernel
attn = CLUSTENQKFunction.apply(query, key, nn_idx)
'''
'''
attn.mean().backward()
grad_query = query.grad.clone().detach()
query.grad.data.zero_()
grad_key = key.grad.clone().detach()
key.grad.data.zero_()

# use the pytorch equivalent
'''
attn2 = CLUSTENQKFunction.apply(query, key, nn_idx)
'''
key_gather = key.gather(index=nn_idx.reshape(b, 1, -1, 1).expand(-1, h, -1, c), dim=2).reshape(b, h, n, m, c)
attn2 = (query.unsqueeze(3) * key_gather).sum(-1)
'''
'''
attn2.mean().backward()
grad_query2 = query.grad.clone().detach()
query.grad.data.zero_()
grad_key2 = key.grad.clone().detach()
key.grad.data.zero_()

print('diff of forward: ', torch.linalg.norm(attn2 - attn))
'''
'''
print('diff of grad query: ', torch.linalg.norm(grad_query2 - grad_query))
print('diff of grad key: ', torch.linalg.norm(grad_key))

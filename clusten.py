###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core
from pathlib import Path

my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind("/")
base_dir = my_dir[:my_len]

clusten_qk_op_lib_path = str(
    next(
        Path(
            next(Path(os.path.join(base_dir, "build")).glob("lib.linux-x86_64-*"))
        ).glob("clusten_qk.cpython-*-x86_64-linux-gnu.so")
    )
)
torch.ops.load_library(clusten_qk_op_lib_path)


class CLUSTENQKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, nbhd_idx):
        # ctx is a context object that can be used to stash information
        # for backward computation
        query = query.contiguous()
        key = key.contiguous()
        if key.dtype != query.dtype:
            key = key.to(query.dtype)
        nbhd_idx = nbhd_idx.contiguous()
        attn = torch.ops.custom_op.clusten_qk(
            query,
            key.permute(0, 1, 3, 2).contiguous(),
            nbhd_idx)
        ctx.save_for_backward(query, key, nbhd_idx)
        return attn

    @staticmethod
    def backward(ctx, grad_attn):
        outputs = torch.ops.custom_op.clusten_qk_backward(grad_attn.contiguous(), *ctx.saved_tensors)
        d_query, d_key = outputs
        return d_query, d_key, None
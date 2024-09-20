#!/usr/bin/env python
# coding=utf-8

import torch
import triton
import triton.language as tl



@triton.jit
def add_vec_kernel(src0_ptr, src1_ptr, dst_ptr, n_elem, BLOCK_SIZE: tl.constexpr):
    # one grogram processes one part of the data
    pid = tl.program_id(axis=0)
    # 
    cur_start_ptr = pid * BLOCK_SIZE
    # offsets = 0, 1, ... BLOCK_SIZE - 1 
    offsets = cur_start_ptr + tl.arange(0, BLOCK_SIZE)
    # avoid out-of-bound memory access   
    mask = offsets < n_elem
    # load x and y from DRAM
    x = tl.load(src0_ptr + offsets, mask=mask)
    y = tl.load(src1_ptr + offsets, mask=mask)
    res = x + y
    # write res to DRAM
    tl.store(dst_ptr + offsets, res, mask=mask)

    
def add(src0 : torch.tensor, src1 : torch.tensor):
    dst = torch.empty_like(src0)
    n_elem = dst.numel()
    # triton.cdiv
    grid = lambda meta : (triton.cdiv(n_elem, meta['BLOCK_SIZE']), )
    add_vec_kernel[grid](src0, src1, dst, n_elem, BLOCK_SIZE=128)
    return dst


if __name__ == "__main__":
    # 
    torch.manual_seed(0)
    vec_size = 11008
    x = torch.rand(vec_size, device='cuda')
    y = torch.rand(vec_size, device='cuda')
    result_torch = x + y
    result_triton = add(x, y)
    max_diff = torch.max(torch.abs(result_triton - result_torch))
    print('max_diff : ', max_diff)
    # 



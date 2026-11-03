import torch
from torch import Tensor
import math
from einops import einsum
from jaxtyping import Float
import triton
import triton.language as tl

def flash_forward_attention_kernel(
    Q: Float[Tensor, " ... nq d"],
    K: Float[Tensor, " ... nk d"],
    V: Float[Tensor, " ... nk d"],
    q_tile_index: int,
    bq: int,
    bk: int,
) -> tuple[Float[Tensor, " ... bq d"], Float[Tensor, " ... bq"]]:
    """
    this kernel is resonsible for a part of output matrix with size (Bq, d).
    Overral, we iterate each K tiles and maintain a running sum and runnming
    max values for each row of the block
    """
    
    # decleare the required local variables and output variables
    device = Q.device
    assert device == K.device and device == V.device
    *_, nq, d = Q.shape
    nk = K.shape[-2]
    assert d == Q.shape[-1]
    assert Q.shape == V.shape

    assert bq > 0 and bk > 0


    batch_shape = Q.shape[:-2]
    l_sum: Float[Tensor, " ... bq"] = torch.zeros((*batch_shape, bq), device=device)
    o: Float[Tensor, " ... bq d"] = torch.zeros((*batch_shape, bq, d), device=device)
    m: Float[Tensor, " ... bq"] = torch.full((*batch_shape, bq), float("-inf"), device=device)

    # load Qi 
    q_row_start = q_tile_index * bq
    q: Float[Tensor, " ... bq d"] = Q[...,q_row_start:q_row_start+bq, :]

    # iterate each k tiles and update running max , running sum and running
    # output sub matrix.
    for j in range(triton.cdiv(nk, bk)):
        # load kj and vj. For pytorch just slicing them from K and V
        k: Float[Tensor, " ... bk d"] = K[...,j*bk:j*bk+bk,:]
        v: Float[Tensor, " ... bk d"] = V[...,j*bk:j*bk+bk,:]

        # compute tile of pre-softmax attention scores s
        s: Float[Tensor, " ... bq bk"] = einsum(
            q,
            k,
            "... bq d, ... bk d -> ... bq bk",
        ) / math.sqrt(d)

        # compute m = max(m_old, rowmax(s)) 
        # we keep the old m to do the rescale later
        m_new: Float[Tensor, " ... bq"] = torch.maximum(s.amax(dim=-1), m)

        # compute the max-value normalized exp value for the new tile
        # p = exp(s - m_new) 
        p: Float[Tensor, " ... bq bk"] = torch.exp(s - m_new[...,None])

        # compute the running sum by appling the rescale
        # l_new = exp(m_new - m_old)*l_old + rowsum(p)
        scale: Float[Tensor, " ... bq"] = torch.exp(m - m_new)
        l_sum = scale * l_sum + torch.sum(p, dim=-1)

        # compute the new o, and we need to apply rescale first
        # scale tensor is expanded as (bq, 1) , then it broadcasts
        # to (bq, bk) by column copies. then element multiply to o
        # since each scale row only effects the same row of matrix o.
        o = scale[...,:,None] * o + einsum(
                p,
                v,
                "... bq bk, ... bk d -> ... bq d",
                )
        m = m_new

    # post re-scale the otuput
    o = o / l_sum[...,None]
    L = m + torch.log(l_sum)
    return o,L


class FlashAttentionForwardPyTorchFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... nq d"],
        K: Float[Tensor, " ... nk d"],
        V: Float[Tensor, " ... nk d"],
        is_causal=False,
    ) -> Float[Tensor, " ... nq d"]:
        bq: int = 32
        bk: int = 32

        *_, nq, d = Q.shape

        q_tiles: int = triton.cdiv(nq, bq)

        o_tiles: list[Float[Tensor, " ... bq d"]] = []
        l_tiles: list[Float[Tensor, " ... bq"]] = []

        for i in range(q_tiles):
            o_tile, l_tile = flash_forward_attention_kernel(
                Q,
                K,
                V,
                i,
                bq,
                bk,
            )
            o_tiles.append(o_tile)
            l_tiles.append(l_tile)

        o = torch.cat(o_tiles, dim=-2)
        l_sum = torch.cat(l_tiles, dim=-1)

        ctx.save_for_backward(l_sum, Q, K, V, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

@triton.jit
def flash_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program indices: each instance covers one batch Q tile (Q_TILE_SIZE, D).
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # initialize block pointers for all matrixes and point them to the
    # start of each of the matrix. Note that we need to skip the batch-dimension stride.
    # Q block: (Q_TILE_SIZE, D),  K block: (K_TILE_SIZE, D)
    # V block: (K_TILE_SIZE, D), L block: (Q_TILE_SIZE,)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + stride_qb*batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + stride_kb*batch_index,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + stride_vb*batch_index,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    # outputs block ptrs
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + stride_ob*batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + stride_lb*batch_index,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # delcare the required local buffers for Li,  Oi and Mi
    # Oi has shape (Q_TILE_SIZE, D), Li has shape (Q_TILE_SIZE,),
    # m has shape (Q_TILE_SIZE,)
    # Note: they all need type as dtype.tl.float32
    Oi = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    Li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    # load Qi from the HBM and will be shared across K tiles
    Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE,D)

    # indices for mask
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # iterate each k tiles and update the running max Mi, running sum Li and running
    # output Oi.
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load Kj and Vj
        Kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)

        # compute tile of pre-softmax atention soces Si
        Si = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # mask out future tokens
            mask = q_indices[:,None] < k_indices[None,:]
            # add a small value for those masked out elements to Si
            Si = tl.where(mask, Si - 1e6, Si)

        # compute Mi_new = max(Mi_old, rowmax(Mi))
        Mi_new = tl.maximum(Mi, tl.max(Si, axis=-1))

        # compute the max-normalized exp value for the new tile
        # Pi = exp(Si - Mi_new)
        Pi = tl.exp(Si - Mi_new[:,None])

        # compute the running sum by applying the rescale
        # Li_new = exp(Mi_old - Mi_new) * Li_old + rowsum(Pi)
        rescale = tl.exp(Mi - Mi_new)
        Li = rescale * Li + tl.sum(Pi, axis=-1)

        # compute the new Oi, and we need to apply rescale as well.
        Oi = tl.dot(Pi.to(Vj.dtype), Vj, acc=rescale[:,None] * Oi)

        # update Mi for next iteration
        Mi = Mi_new

        # advance K,V block ptrs
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # post re-scale the output Oi
    Oi = Oi / Li[:,None]
    # store back to the HBM for both Oi and Li
    Lse = Mi + tl.log(Li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, Lse.to(L_block_ptr.type.element_ty), boundary_check=(0,))

class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        out = torch.empty((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=Q.dtype)
        scale: float = 1.0 / math.sqrt(D)

        flash_forward_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q,K,V,
            out, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, out)
        return out


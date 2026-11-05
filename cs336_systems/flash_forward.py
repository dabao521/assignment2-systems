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

# return dQ, dK, dV
@torch.compile
def flash_attention_backward_pytorch(
    Q: Float[Tensor, " ... nq d"],
    K: Float[Tensor, " ... nk d"],
    V: Float[Tensor, " ... nk d"],
    Out: Float[Tensor, " ... nq d"],
    dO: Float[Tensor, " ... nq d"],
    L: Float[Tensor, " ... nq"]
) -> tuple[
    Float[Tensor, " ... nq d"],
    Float[Tensor, " ... nk d"],
    Float[Tensor, " ... nk d"]
    ]:
    """
    Now following the equation 13-19 to compute the QKV derivatives
    """
    d: int = Q.shape[-1]
    scale: float = 1.0 / math.sqrt(d)

    # Equation 13: compute activation S
    S: Float[Tensor, " ... nq nk"] = einsum(
        Q,
        K,
        "... nq d, ... nk d -> ... nq nk",
    ) * scale

    # Equation 14: compute activation P after softmax on S
    P: Float[Tensor, " ... nq nk"] = torch.exp(S - L[...,None])

    # Equation 15: now coppute dV
    dV: Float[Tensor, " ... nk d"] = einsum(
        P,
        dO,
        "... nq nk, ... nq d -> ... nk d",
    )

    # Equation 16: compute dP backward matrix for next steps
    dP: Float[Tensor, " ... nq nk"] = einsum(
        dO,
        V,
        "... nq d, ... nk d -> ... nq nk"
    )

    # Equation 17: compute dS
    # D = rowsum(O * dO)
    D: Float[Tensor, " ... nq"] = torch.sum(Out * dO, dim=-1)
    dS: Float[Tensor, " ... nq nk"] =  P * (dP - D[...,None])

    # Equation 18: compute dQ
    dQ: Float[Tensor, " ... nq d"] = einsum(
        dS,
        K,
        "... nq nk, ... nk d -> ... nq d",
    ) * scale

    # Equation 19 : compute dK
    dK: Float[Tensor, " ... nk d"] = einsum(
        dS,
        Q,
        "... nq nk, ... nq d -> ... nk d",
    ) * scale

    return dQ, dK, dV
    


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
        # load the stored input matrix from the forward pass.
        L, Q, K, V, Out = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward_pytorch(Q,K,V,Out,grad_out,L)
        return dQ, dK, dV, None

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

@triton.jit
def flash_backward_triton_kv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, # inputs
    dK_ptr, dV_ptr, # outputs
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program indices
    k_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # inputs block ptrs
    Kj_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    Vj_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    Qi_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    dOi_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index*stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index*stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # outpus block ptrs
    dKj_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index*stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    dVj_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index*stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    # load Kj and Vj
    Kj = tl.load(Kj_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE,D)
    Vj = tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE,D)
    

    # initialize shared buffer dKj and dVj which will be written back
    # to global memory
    dKj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32) # (K_TILE_SIZE,D)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32) # (K_TILE_SIZE,D)

    k_indices = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    # iterate each Q tile
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # load Qi, dOi, Di, Li
        Qi = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE, D)
        dOi = tl.load(dOi_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE, D)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) # (Q_TILE_SIZE,)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) # (Q_TILE_SIZE,)

        Si = tl.dot(Qi, tl.trans(Kj)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            q_indices = i*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_indices[:,None] < k_indices[None,:]
            Si = tl.where(mask, Si - 1e6, Si)

        Pi = tl.exp(Si - Li[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dVj = tl.dot(tl.trans(Pi.to(dOi.dtype)), dOi, acc=dVj) # (K_TILE_SIZE, D)
        dPi = tl.dot(dOi, tl.trans(Vj)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dSi = Pi * (dPi - Di[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dKj = dKj + tl.dot(tl.trans(dSi.to(Qi.dtype)), Qi) * scale # (K_TILE_SIZE, D)

        # advance Qi, dOi, Di, Li block pointers
        Qi_block_ptr = Qi_block_ptr.advance((Q_TILE_SIZE, 0))
        dOi_block_ptr = dOi_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))

    # write back dKj and dVj back to HBM
    tl.store(dKj_block_ptr, dKj.to(dKj_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(dVj_block_ptr, dVj.to(dVj_block_ptr.type.element_ty), boundary_check=(0,1))

@triton.jit
def flash_backward_triton_q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, # inputs
    dQ_ptr, # outputs
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program indices
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # inputs block ptrs
    Qi_block_ptr = tl.make_block_ptr(
        base=Q_ptr + stride_qb*batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    dOi_block_ptr = tl.make_block_ptr(
        base=dO_ptr + stride_qb*batch_index,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    Vi_block_ptr = tl.make_block_ptr(
        base=V_ptr + stride_vb*batch_index,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    Ki_block_ptr = tl.make_block_ptr(
        base=K_ptr + stride_kb*batch_index,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    Li_block_ptr = tl.make_block_ptr(
        base=L_ptr + stride_lb*batch_index,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Di_block_ptr = tl.make_block_ptr(
        base=D_ptr + stride_db*batch_index,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # outputs block ptrs
    dQi_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + stride_dqb*batch_index,
        shape=(N_QUERIES,D),
        strides=(stride_dqq,stride_dqd),
        offsets=(q_tile_index*Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )

    # load Qi, dOi
    Qi = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE, D)
    dOi = tl.load(dOi_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE, D)
    Li = tl.load(Li_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) # (Q_TILE_SIZE,)
    Di = tl.load(Di_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32) # (Q_TILE_SIZE,)

    # initialize dQi
    dQi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_indices = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # for each K tile doing the backward process
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load Kj and Vj
        Kj = tl.load(Ki_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(Vi_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)

        Si = tl.dot(Qi, tl.trans(Kj)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            k_indices = j*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:,None] < k_indices[None,:]
            Si = tl.where(mask, Si - 1e6, Si)

        Pi = tl.exp(Si - Li[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dPi = tl.dot(dOi, tl.trans(Vj)) # (Q_TILE_SIZE, K_TILE_SIZE)
        dSi = Pi * (dPi - Di[:,None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        dQi = dQi + tl.dot(dSi.to(Kj.dtype), Kj) * scale # (Q_TILE_SIZE, D)

        # advance pointers
        Ki_block_ptr = Ki_block_ptr.advance((K_TILE_SIZE,0))
        Vi_block_ptr = Vi_block_ptr.advance((K_TILE_SIZE,0))

    # write back to the dQ HBM
    tl.store(dQi_block_ptr, dQi.to(dQi_block_ptr.type.element_ty), boundary_check=(0,1))

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
        ctx.is_causal = is_causal
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        L,Q,K,V,Out = ctx.saved_tensors
        batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # compute Dvs for backward
        Dvs = torch.sum(Out * grad_out, dim=-1) # (batch_size, N_QUERIES)

        scale: float = 1.0 / math.sqrt(D)

        # kernel1 
        # initialize dK and dV buffer
        dK = torch.empty((batch_size, N_KEYS, D), device=K.device, dtype=K.dtype)
        dV = torch.empty((batch_size, N_KEYS, D), device=V.device, dtype=V.dtype)

        flash_backward_triton_kv_kernel[(triton.cdiv(N_KEYS, K_TILE_SIZE), batch_size)](
            Q,K,V,grad_out,L,Dvs,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            L.stride(0), L.stride(1),
            Dvs.stride(0), Dvs.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            ctx.is_causal,
        )

        # kerne2
        # initialize dQ buffer
        dQ = torch.empty((batch_size, N_QUERIES, D), device=Q.device, dtype=Q.dtype)

        flash_backward_triton_q_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q,K,V,grad_out,L,Dvs,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            L.stride(0), L.stride(1),
            Dvs.stride(0), Dvs.stride(1),
            dQ.stride(0), dK.stride(1), dK.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            ctx.is_causal,
        )

        return dQ, dK, dV, None


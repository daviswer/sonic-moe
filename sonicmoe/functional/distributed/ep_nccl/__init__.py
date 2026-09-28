# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
# Rank-local, undeduped NCCL transport for ``DispatchMode.A2A_NCCL`` /
# ``CombineMode.A2A_NCCL``. Viable across nodes, where symm-mem (NVLink/P2P
# -only) is not.
#
# Unlike RANK_DEDUP (whose metadata is derived from a global, all-gathered
# ``topk_idx``), this path needs no topology-wide collective at all: each
# rank computes its own outgoing routing from its own local ``topk_idx``,
# learns how many rows it will receive via one tiny ``(W,)``-sized
# ``all_to_all_single`` (replacing the global all-gather entirely), and
# transports every (token, expert) slot as its own row -- no dedup. The
# per-token K-way reduction that combine needs happens *after* the reverse
# transport lands data home, as a plain local sum -- there is no
# ``pair_present_mask``/``local_combine`` kernel in this path, since every
# slot's contribution arrives as a distinct row.
#
# Send-side ordering: ``argsort(dst_rank_flat, stable=True)`` groups slots by
# destination while preserving ascending-slot (hence ascending-token) order
# within each destination's chunk -- ``all_to_all_single``'s own placement
# convention (rank p's chunk lands in ascending-p order, in send order within
# a chunk) then reproduces a clean per-source-rank, ascending-order receive
# layout with no further bookkeeping.
# ********************************************************************************

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def compute_local_routing(topk_idx_local: torch.Tensor, E_local: int, W: int) -> dict:
    """Pure-local routing decision -- no communication.

    Args:
        topk_idx_local: (T_local, K) int -- global expert ids picked by this
            rank's own tokens. Global id ``e`` maps to destination rank
            ``e // E_local`` and local expert ``e % E_local`` (block layout,
            matching ``compute_dispatch_metadata``'s convention).
        E_local: experts per rank.
        W: EP world size.

    Returns a dict with:
        dst_rank_flat: (TK_local,) int32 -- destination rank per slot.
        local_expert_flat: (TK_local,) int32 -- local expert id per slot.
        send_order: (TK_local,) int64 -- permutation of slot indices, stable-
            sorted by destination rank.
        send_splits_local: (W,) int32 -- per-destination outgoing slot count.
    """
    flat = topk_idx_local.reshape(-1).to(torch.int64)
    dst_rank_flat = torch.div(flat, E_local, rounding_mode="floor").to(torch.int32)
    local_expert_flat = (flat - dst_rank_flat.to(torch.int64) * E_local).to(torch.int32)
    send_order = torch.argsort(dst_rank_flat, stable=True)
    send_splits_local = torch.bincount(dst_rank_flat, minlength=W).to(torch.int32)
    return {
        "dst_rank_flat": dst_rank_flat,
        "local_expert_flat": local_expert_flat,
        "send_order": send_order,
        "send_splits_local": send_splits_local,
    }


def exchange_split_counts(send_splits_local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """One tiny ``(W,)``-sized ``all_to_all_single`` to learn per-source receive
    counts -- the only collective this path needs to discover shapes, replacing
    RANK_DEDUP's global ``topk_idx`` all-gather entirely."""
    recv_splits_local = torch.empty_like(send_splits_local)
    dist.all_to_all_single(recv_splits_local, send_splits_local, group=group)
    return recv_splits_local


def nccl_a2a(
    send: torch.Tensor,
    send_splits: list,
    recv_splits: list,
    group: dist.ProcessGroup,
    pad_to: Optional[int] = None,
) -> torch.Tensor:
    """Thin ``all_to_all_single`` wrapper, reused for every payload this path
    moves: the forward's X / scores / local-expert-id dispatch, and the
    backward's dO dispatch and dx / ds reverse-combine (with ``send_splits``
    and ``recv_splits`` swapped by the caller for the reverse direction).

    ``pad_to``: when given (>= n_recv), allocate the receive buffer at this
    row count instead of the tight ``n_recv`` and write the transport's real
    rows into its ``[:n_recv]`` prefix, returning the full padded tensor. Use
    this for any payload that becomes a GEMM's own tensor argument (e.g. the
    up-proj GEMM's ``A``) -- quack's autotuner keys its config cache on every
    tensor argument's shape (not just the declared ``key=`` fields), so a
    tightly-sized, routing-dependent shape busts that cache on every call and
    re-triggers a full autotune sweep. Payloads that only feed a plain,
    non-autotuned kernel (or no kernel at all) don't need this.
    """
    n_recv = sum(recv_splits)
    trailing = tuple(send.shape[1:])
    if pad_to is None:
        recv = torch.empty((n_recv,) + trailing, dtype=send.dtype, device=send.device)
        dist.all_to_all_single(recv, send, output_split_sizes=recv_splits, input_split_sizes=send_splits, group=group)
        return recv
    recv = torch.empty((pad_to,) + trailing, dtype=send.dtype, device=send.device)
    dist.all_to_all_single(
        recv[:n_recv], send, output_split_sizes=recv_splits, input_split_sizes=send_splits, group=group
    )
    return recv


def reorder_by_send_order(x_local: torch.Tensor, send_order: torch.Tensor, K: int) -> torch.Tensor:
    """Gather this rank's per-token rows into per-slot send order (one row per
    (token, expert) slot, K-way duplicated per token -- undeduped)."""
    send_token_idx = torch.div(send_order, K, rounding_mode="floor")
    return x_local.index_select(0, send_token_idx)


def unpermute_and_reduce(
    received: torch.Tensor,
    send_order: torch.Tensor,
    T_local: int,
    K: int,
    reduce: bool,
) -> torch.Tensor:
    """Reverse-combine's final local step: un-permute the reverse-transported
    (TK_local, ...) buffer back into original (t, k) slot order via
    ``send_order``'s inverse, then optionally sum over K (dx) or just reshape
    (ds)."""
    trailing = received.shape[1:]
    buf = torch.empty((T_local * K,) + trailing, dtype=received.dtype, device=received.device)
    buf[send_order] = received
    buf = buf.view((T_local, K) + trailing)
    return buf.sum(dim=1) if reduce else buf


def scatter_grouped_to_received(
    grouped: torch.Tensor,
    x_gather_idx: torch.Tensor,
    n_recv: int,
) -> torch.Tensor:
    """Un-group a (n_recv, ...) grouped-by-expert buffer back into received-row
    order. ``x_gather_idx[:n_recv]`` is a permutation under undeduped transport
    (bijection grouped-position -> received-row-index), so this scatter is
    total and well-defined."""
    out = torch.empty((n_recv,) + tuple(grouped.shape[1:]), dtype=grouped.dtype, device=grouped.device)
    out[x_gather_idx[:n_recv].to(torch.int64)] = grouped[:n_recv]
    return out

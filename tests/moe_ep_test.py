# ********************************************************************************
# Copyright (c) 2026, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# Multi-rank correctness test for the EP forward + backward.
#
# Each rank constructs the same MoE (seeded), shards weights along the leading
# E axis, runs the EP forward AND backward on its slice of the global x,
# all-gathers the per-rank outputs and gradients, and compares against a
# single-rank PyTorch autograd reference computed on rank 0.
#
# Sweep: bias × routing-variant × {3 dispatch modes × 3 combine modes}
#        × {TC_softmax_topk (fwd + bwd), general_routing (fwd only)}
# General routing's fwd+bwd combination currently trips a QuACK DSL
# compile-time ICE in epi_ops.cute.copy on the test's smaller shapes —
# tracked in QuACK, not in EP. TC covers the backward sweep.
#
# Usage (torchrun-launched):
#
#   torchrun --nproc_per_node=4 --standalone --local-ranks-filter 0 \
#            tests/moe_ep_test.py
#   torchrun --nproc_per_node=8 --standalone --local-ranks-filter 0 \
#            tests/moe_ep_test.py --concat-layout
#
# --ep-world-size splits WORLD_SIZE ranks into WORLD_SIZE/ep-world-size independent
# EP groups of ep_world_size ranks each (contiguous rank blocks), so this same driver
# also covers the hybrid `ep < dps` (fsdp_ep > 1) topology: each group runs the full
# fp32-oracle check as if it were its own standalone EP world, using its own
# dist.new_group rather than the default WORLD group for every collective. Implicitly
# forces --only-nccl (see that flag's help) since a hybrid group's ranks are placed to
# be cross-node by construction. E.g. 2 EP groups of 2 across 4 nodes:
#
#   torchrun --nnodes=4 --nproc_per_node=1 ... tests/moe_ep_test.py --ep-world-size 2
# ********************************************************************************

from __future__ import annotations

import argparse
import datetime
import itertools
import os
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import quack.autotuner
import torch


# !!!!!!!! sometimes there is a broken pipe issue when QuACK compile worker > 1 !!!!!!!!
os.environ["QUACK_COMPILE_WORKERS"] = "1"


# ─────────────── Monkey-patch: similar M shapes map to the same cached config during QuACK autotuning ───────────────
# Mirrors ``benchmarks/distributed/moe-ep.py``: quantizing M in the autotune cache key
# stops re-tuning on every shape, and the same restricted config set keeps both files
# consistent.

M_QUANT = 1024


def _make_quantized_key(self, args, kwargs):
    all_args = {**dict(zip(self.arg_names, args)), **kwargs}
    _args = {k: v for k, v in all_args.items() if k in self.arg_names}
    key = [str(_args[k]) for k in self.keys if k in _args]
    for _, arg in _args.items():
        if isinstance(arg, torch.Tensor):
            s = list(arg.shape)
            if s and s[0] >= M_QUANT:
                s[0] = ((s[0] + M_QUANT - 1) // M_QUANT) * M_QUANT
            key.append(str(tuple(s)))
            key.append(str([x if x in {0, 1} else 2 for x in arg.stride()]))
            key.append(str(arg.dtype))
    return tuple(key)


_orig_call = quack.autotuner.Autotuner.__call__


@torch.compiler.disable
def _patched_call(self, *args, **kwargs):
    if len(self.configs) > 1:
        qkey = _make_quantized_key(self, args, kwargs)
        if qkey in self.cache:
            config = self.cache[qkey]
            self.best_config = config
            self.nargs = dict(zip(self.arg_names, args))
            ret = self.fn.__call__(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret
    ret = _orig_call(self, *args, **kwargs)
    if len(self.configs) > 1 and hasattr(self, "best_config"):
        qkey = _make_quantized_key(self, args, kwargs)
        self.cache[qkey] = self.best_config
    return ret


quack.autotuner.Autotuner.__call__ = _patched_call


# ─────────────── Monkey-patch: restrict SM100 / SM90 autotuning to the same set the bench uses ───────────────

import quack.gemm_config as _gc
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_dgated_tuned, gemm_gated_tuned, gemm_tuned


def _fast_sm100_configs(epilogue=None):
    tile_n_vals = [128, 192, 256]
    tile_mn_cluster_vals = [(256, tile_n, (2, 1)) for tile_n in tile_n_vals] + [(256, 512, (2, 1))]
    GemmConfigCls = partial(GemmConfig, pingpong=False, device_capacity=10)
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=False,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


def _fast_sm90_configs(epilogue=None, tune_coop=True):
    tile_n_vals = [128, 160, 192]
    tile_mn_vals_coop = [(256, tile_n) for tile_n in tile_n_vals] + [(128, 256)]
    tile_mn_vals_pingpong = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["gated"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if n % 32 == 0 and m != 192]
        tile_mn_vals_pingpong = [(m, n) for m, n in tile_mn_vals_pingpong if n % 32 == 0]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            device_capacity=9,
            is_dynamic_persistent=False,
            use_tma_gather=False,
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
        )
    ]


_gc._get_sm100_configs = _fast_sm100_configs
_gc._get_sm90_configs = _fast_sm90_configs


def _patch_autotuner_configs(autotuner_fn):
    autotuner_fn.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]


_patch_autotuner_configs(gemm_tuned)
_patch_autotuner_configs(gemm_gated_tuned)
_patch_autotuner_configs(gemm_dgated_tuned)
gemm_gated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]
gemm_dgated_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs("gated")]


import torch.distributed as dist
import torch.nn.functional as F

from sonicmoe import MoE
from sonicmoe.distributed_utils import CombineMode, DispatchMode, RuntimeEPConfig  # type: ignore
from sonicmoe.enums import ActivationType
from sonicmoe.functional import TC_Softmax_Topk_Router_Function
from sonicmoe.functional.ep import _row_bins, moe_ep_general_routing_forward, moe_ep_TC_softmax_topk_forward


@dataclass
class Shape:
    name: str
    T: int
    H: int
    I: int
    E: int
    K: int


SHAPES: List[Shape] = [
    # K spans K < W, K == W and K > W at the W=4 single-node default; K_eq_4 is the topology the
    # training configs use (E=48, K=4 -> E_local=12), which the sweep originally skipped.
    Shape("K_eq_2", T=4096, H=2048, I=1024, E=32, K=2),
    Shape("K_eq_4", T=4096, H=2048, I=1024, E=48, K=4),
    Shape("K_eq_8", T=4096, H=2048, I=1024, E=64, K=8),
    Shape("K_eq_10", T=4096, H=2048, I=512, E=64, K=10),
]


ROUTING_VARIANTS: List[Tuple[str, bool, bool]] = [
    # (name, is_softmax_over_topk, norm_topk_probs)
    ("topk_then_softmax_norm", False, True),
]


# Full 3 × 3 sweep of dispatch × combine primitives.
DISPATCH_MODES: List[DispatchMode] = [
    DispatchMode.AG_DISPATCH_TRITON,
    DispatchMode.A2A_DISPATCH_TRITON,
    DispatchMode.RANK_DEDUP_DISPATCH_TRITON,
]
COMBINE_MODES: List[CombineMode] = [
    CombineMode.A2A_COMBINE_TRITON,
    CombineMode.RS_COMBINE_TRITON,
    CombineMode.RANK_DEDUP_COMBINE_TRITON,
]

# The 3x3 sweep above is a free cartesian product -- combine mode is orthogonal
# to dispatch mode for all three symm-mem pairs. A2A_NCCL is not: dispatch_mode
# and combine_mode must both be A2A_NCCL together (see _validate_runtime_ep_config
# -- mixing it with a symm-mem mode would attempt a symm-mem rendezvous on a
# cross-node group and hang), so it's tested as one explicit extra pair rather
# than added to the cartesian sweep above.
DISPATCH_COMBINE_MODE_PAIRS: List[Tuple[DispatchMode, CombineMode]] = [
    (d, c) for d in DISPATCH_MODES for c in COMBINE_MODES
] + [(DispatchMode.A2A_NCCL, CombineMode.A2A_NCCL)]


# ============================================================================
# Helpers
# ============================================================================


def _swiglu(h: torch.Tensor, concat_layout: bool = False) -> torch.Tensor:
    if concat_layout:
        g, u = torch.chunk(h, 2, dim=-1)
    else:
        u, g = h[..., 1::2], h[..., ::2]
    return u * F.silu(g)


def _all_gather_y(
    y_local: torch.Tensor, world_size: int, group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-gather a (T_local, *trailing) tensor along dim 0 into (W*T_local, *trailing),
    scoped to `group` (None = the default WORLD group)."""
    out_shape = (world_size * y_local.shape[0],) + tuple(y_local.shape[1:])
    out = torch.empty(out_shape, dtype=y_local.dtype, device=y_local.device)
    dist.all_gather_into_tensor(out, y_local.contiguous(), group=group)
    return out


def _gather_to_rank0(
    t_local: torch.Tensor, world_size: int, rank: int, group: Optional[dist.ProcessGroup] = None
) -> Optional[List[torch.Tensor]]:
    """Gather t_local from every rank in `group` into a list of W tensors on that
    group's local rank 0 (`rank` is already group-relative)."""
    t_local = t_local.contiguous()
    out = torch.empty((world_size,) + tuple(t_local.shape), dtype=t_local.dtype, device=t_local.device)
    dist.all_gather_into_tensor(out, t_local, group=group)
    if rank == 0:
        return list(out.unbind(dim=0))
    return None


def _strided_clone(t: torch.Tensor) -> torch.Tensor:
    """Preserve the non-contiguous strided layout that quack's grouped GEMM
    requires for ``w1`` (``(2I, H, E_local)`` view of the original contiguous
    ``(E_local, 2I, H)`` backing). A plain ``.clone()`` would reset strides to
    the contiguous layout for the new shape and trip the kernel's stride
    check."""
    out = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    return out.copy_(t)


# ============================================================================
# References — single-rank PyTorch autograd in fp32, used for fwd+bwd compare.
# ============================================================================


def _per_expert_reference_tc(
    x_global: torch.Tensor,
    router_w: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    b1_full: Optional[torch.Tensor],
    b2_full: Optional[torch.Tensor],
    topk_idx_global: torch.Tensor,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    concat_layout: bool,
    dout_global: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """TC reference: per-expert MoE forward + autograd backward in fp32.

    Uses ``topk_idx_global`` from the EP path so EP and the reference agree
    on which experts each token routes to (TC's topk tie-breaking is its
    own kernel — driving the reference off PyTorch's ``logits.topk`` would
    diverge under ties and produce incomparable backward grads).
    """
    ref_x = x_global.detach().to(torch.float32).requires_grad_(True)
    ref_router_w = router_w.detach().to(torch.float32).requires_grad_(True)
    ref_w1 = w1_full.detach().to(torch.float32).requires_grad_(True)
    ref_w2 = w2_full.detach().to(torch.float32).requires_grad_(True)
    ref_b1 = b1_full.detach().to(torch.float32).requires_grad_(True) if b1_full is not None else None
    ref_b2 = b2_full.detach().to(torch.float32).requires_grad_(True) if b2_full is not None else None

    logits = F.linear(ref_x, ref_router_w)
    if is_softmax_over_topk:
        topk_logits = torch.gather(logits, 1, topk_idx_global)
        topk_scores = topk_logits.softmax(dim=-1)
    else:
        probs = logits.softmax(dim=-1)
        topk_scores = torch.gather(probs, 1, topk_idx_global)
        if norm_topk_probs:
            topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    T, H = ref_x.shape
    E = ref_w1.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=ref_x.device)
    for i in range(E):
        rows_t, rows_k = (topk_idx_global == i).nonzero(as_tuple=True)
        if rows_t.numel() == 0:
            continue
        h = F.linear(
            ref_x[rows_t],
            ref_w1[i],
            bias=(ref_b1[i] if ref_b1 is not None else None),
        )
        h = _swiglu(h, concat_layout=concat_layout)
        y = F.linear(
            h,
            ref_w2[i],
            bias=(ref_b2[i] if ref_b2 is not None else None),
        )
        contrib = y * topk_scores[rows_t, rows_k, None]
        out = out.index_add(0, rows_t, contrib)

    inputs = [ref_x, ref_router_w, ref_w1, ref_w2]
    if ref_b1 is not None:
        inputs += [ref_b1, ref_b2]
    grads = torch.autograd.grad(out, inputs, grad_outputs=dout_global.to(torch.float32))

    ref_dx, ref_drouter_w, ref_dw1, ref_dw2 = grads[:4]
    ref_db1 = grads[4] if ref_b1 is not None else None
    ref_db2 = grads[5] if ref_b1 is not None else None
    return (
        out.detach(),
        ref_dx.detach(),
        ref_drouter_w.detach(),
        ref_dw1.detach(),
        ref_dw2.detach(),
        ref_db1.detach() if ref_db1 is not None else None,
        ref_db2.detach() if ref_db2 is not None else None,
    )


def _per_expert_reference_general(
    x_global: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    b1_full: Optional[torch.Tensor],
    b2_full: Optional[torch.Tensor],
    topk_idx_global: torch.Tensor,
    topk_scores_global: torch.Tensor,
    concat_layout: bool,
    dout_global: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """General-routing reference: per-expert MoE forward + autograd backward in fp32.

    Routing is caller-supplied, so there is no router to differentiate — the grad of the
    supplied scores (``ds``) takes ``drouter_w``'s place."""
    ref_x = x_global.detach().to(torch.float32).requires_grad_(True)
    ref_scores = topk_scores_global.detach().to(torch.float32).requires_grad_(True)
    ref_w1 = w1_full.detach().to(torch.float32).requires_grad_(True)
    ref_w2 = w2_full.detach().to(torch.float32).requires_grad_(True)
    ref_b1 = b1_full.detach().to(torch.float32).requires_grad_(True) if b1_full is not None else None
    ref_b2 = b2_full.detach().to(torch.float32).requires_grad_(True) if b2_full is not None else None

    T, H = ref_x.shape
    E = ref_w1.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=ref_x.device)
    for i in range(E):
        rows_t, rows_k = (topk_idx_global == i).nonzero(as_tuple=True)
        if rows_t.numel() == 0:
            continue
        h = F.linear(
            ref_x[rows_t],
            ref_w1[i],
            bias=(ref_b1[i] if ref_b1 is not None else None),
        )
        h = _swiglu(h, concat_layout=concat_layout)
        y = F.linear(
            h,
            ref_w2[i],
            bias=(ref_b2[i] if ref_b2 is not None else None),
        )
        contrib = y * ref_scores[rows_t, rows_k, None]
        out = out.index_add(0, rows_t, contrib)

    inputs = [ref_x, ref_scores, ref_w1, ref_w2]
    if ref_b1 is not None:
        inputs += [ref_b1, ref_b2]
    grads = torch.autograd.grad(out, inputs, grad_outputs=dout_global.to(torch.float32))

    ref_dx, ref_ds, ref_dw1, ref_dw2 = grads[:4]
    ref_db1 = grads[4] if ref_b1 is not None else None
    ref_db2 = grads[5] if ref_b1 is not None else None
    return (
        out.detach(),
        ref_dx.detach(),
        ref_ds.detach(),
        ref_dw1.detach(),
        ref_dw2.detach(),
        ref_db1.detach() if ref_db1 is not None else None,
        ref_db2.detach() if ref_db2 is not None else None,
    )


# ============================================================================
# Comparison
# ============================================================================


def _check_quantities(
    tag: str,
    log_prefix: str,
    quantities: List[Tuple[str, torch.Tensor, torch.Tensor]],
    atol: float,
    rtol: float,
) -> Tuple[bool, str]:
    """Element-wise compare each (ep, ref) pair with assert_close. Annotate
    pass/fail per quantity with max-abs / mean-rel stats for diagnostics."""
    msgs = []
    ok_all = True
    for name, ep, ref in quantities:
        ep_f = ep.float()
        ref_f = ref.float()
        diff = (ep_f - ref_f).abs()
        max_d = diff.max().item()
        mean_r = (diff / (ref_f.abs() + 1e-6)).mean().item()
        try:
            torch.testing.assert_close(ep_f, ref_f, atol=atol, rtol=rtol)
            msgs.append(f"{name}:OK")
        except AssertionError:
            ok_all = False
            # ep/ref over the significant entries: a flat ratio means a pure scale factor
            # (e.g. a missing 1/W), a spread means the mismatch is structural
            sig = (ref_f.abs() > ref_f.abs().max() * 1e-3).nonzero().flatten()
            if sig.numel() > 1 << 20:
                sig = sig[torch.randperm(sig.numel(), device=sig.device)[: 1 << 20]]
            ratio_txt = ""
            if sig.numel() > 0:
                r = (ep_f.flatten()[sig] / ref_f.flatten()[sig]).float()
                q = torch.quantile(r, torch.tensor([0.01, 0.5, 0.99], device=r.device))
                ratio_txt = f" ep/ref p1={q[0]:.4f} p50={q[1]:.4f} p99={q[2]:.4f}"
            msgs.append(f"{name}:FAIL[max={max_d:.2e} rel={mean_r:.2e}{ratio_txt}]")
    head = f"{log_prefix}{tag}"
    marker = "✓ PASS" if ok_all else "✗ FAIL"
    return ok_all, f"{head:<88s} {marker}  " + "  ".join(msgs)


# ============================================================================
# Per-config EP runners (forward + backward, gather grads to rank 0).
# ============================================================================


def _run_ep_tc_one(
    x_local: torch.Tensor,
    router_w: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    b1_local: Optional[torch.Tensor],
    b2_local: Optional[torch.Tensor],
    dout_local: torch.Tensor,
    K: int,
    E: int,
    cfg: RuntimeEPConfig,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    concat_layout: bool,
    world_size: int,
    rank: int,
    group: Optional[dist.ProcessGroup] = None,
    CPU_sync_on_runtime: bool = False,
):
    """Run one TC EP fwd+bwd; gather per-rank grads to rank 0. Returns the
    (y_full, dx_full, drouter_w, ep_dw1, ep_dw2, ep_db1, ep_db2) tuple on
    rank 0; returns ``None`` on other ranks."""
    x_t = x_local.detach().clone().requires_grad_(True)
    router_w_t = router_w.detach().clone().requires_grad_(True)
    w1_t = _strided_clone(w1_local).requires_grad_(True)
    w2_t = w2_local.detach().clone().requires_grad_(True)
    b1_t = b1_local.detach().clone().requires_grad_(True) if b1_local is not None else None
    b2_t = b2_local.detach().clone().requires_grad_(True) if b2_local is not None else None

    # moe_ep_TC_softmax_topk_forward returns (out, router_logits, expert_frequency);
    # this test compares only the MoE output + its grads.
    y_local, _router_logits, _expert_freq = moe_ep_TC_softmax_topk_forward(
        x_t,
        router_w_t,
        w1_t,
        b1_t,
        w2_t,
        b2_t,
        K=K,
        E=E,
        group=group,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        is_softmax_over_topk=is_softmax_over_topk,
        norm_topk_probs=norm_topk_probs,
        concat_layout=concat_layout,
        ep_config=cfg,
        CPU_sync_on_runtime=CPU_sync_on_runtime,
    )

    inputs = [x_t, router_w_t, w1_t, w2_t]
    if b1_t is not None:
        inputs += [b1_t, b2_t]
    grads = torch.autograd.grad(y_local, inputs, grad_outputs=dout_local, retain_graph=False)

    dx_local, drouter_w_local, dw1_local, dw2_local = grads[:4]
    db1_local = grads[4] if b1_t is not None else None
    db2_local = grads[5] if b1_t is not None else None

    # For router_w replicated across ranks (and running on local tokens),
    # reduce the grad manually (training code handles this e.g. FSDP)
    drouter_w_local = drouter_w_local.contiguous()
    dist.all_reduce(drouter_w_local, op=dist.ReduceOp.SUM, group=group)

    y_full = _all_gather_y(y_local.detach(), world_size, group)
    dx_full = _all_gather_y(dx_local, world_size, group)
    dw1_list = _gather_to_rank0(dw1_local, world_size, rank, group)
    dw2_list = _gather_to_rank0(dw2_local, world_size, rank, group)
    db1_list = _gather_to_rank0(db1_local, world_size, rank, group) if db1_local is not None else None
    db2_list = _gather_to_rank0(db2_local, world_size, rank, group) if db2_local is not None else None

    if rank == 0:
        # dw1 per rank is (2I, H, E_local) — concat along E_local axis (dim 2)
        # to reconstruct (2I, H, E). dw2 per rank is (E_local, I, H) — concat
        # along dim 0 to reconstruct (E, I, H).
        ep_dw1 = torch.cat(dw1_list, dim=2)
        ep_dw2 = torch.cat(dw2_list, dim=0)
        ep_db1 = torch.cat(db1_list, dim=0) if db1_list is not None else None
        ep_db2 = torch.cat(db2_list, dim=0) if db2_list is not None else None
        return y_full, dx_full, drouter_w_local.detach(), ep_dw1, ep_dw2, ep_db1, ep_db2
    return None


def _run_ep_general_one_fwd(
    x_local: torch.Tensor,
    idx_local: torch.Tensor,
    scores_local: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    b1_local: Optional[torch.Tensor],
    b2_local: Optional[torch.Tensor],
    E: int,
    cfg: RuntimeEPConfig,
    concat_layout: bool,
    world_size: int,
    rank: int,
    group: Optional[dist.ProcessGroup] = None,
    CPU_sync_on_runtime: bool = False,
):
    """Run general-routing EP forward in inference mode and return y_full
    on rank 0. Forward-only is enough to exercise the 3×3 dispatch×combine
    matrix on the general entry point; TC covers the backward sweep."""
    w1_no_grad = _strided_clone(w1_local)
    # moe_ep_general_routing_forward returns (out, expert_frequency).
    y_local, _expert_freq = moe_ep_general_routing_forward(
        x_local,
        idx_local,
        scores_local,
        w1_no_grad,
        b1_local,
        w2_local,
        b2_local,
        E=E,
        group=group,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
        concat_layout=concat_layout,
        ep_config=cfg,
        CPU_sync_on_runtime=CPU_sync_on_runtime,
    )
    y_full = _all_gather_y(y_local.detach(), world_size, group)
    if rank == 0:
        return y_full


def _run_ep_general_one_train(
    x_local: torch.Tensor,
    idx_local: torch.Tensor,
    scores_local: torch.Tensor,
    w1_local: torch.Tensor,
    w2_local: torch.Tensor,
    b1_local: Optional[torch.Tensor],
    b2_local: Optional[torch.Tensor],
    dout_local: torch.Tensor,
    E: int,
    cfg: RuntimeEPConfig,
    concat_layout: bool,
    world_size: int,
    rank: int,
    group: Optional[dist.ProcessGroup] = None,
    CPU_sync_on_runtime: bool = False,
):
    """Run one general-routing EP fwd+bwd in training mode; gather per-rank grads to
    rank 0. Returns (y_full, dx_full, ds_full, ep_dw1, ep_dw2, ep_db1, ep_db2) on rank 0,
    ``None`` elsewhere. The pre-flight arm calls this and discards the result."""
    x_t = x_local.detach().clone().requires_grad_(True)
    scores_t = scores_local.detach().clone().requires_grad_(True)
    w1_t = _strided_clone(w1_local).requires_grad_(True)
    w2_t = w2_local.detach().clone().requires_grad_(True)
    b1_t = b1_local.detach().clone().requires_grad_(True) if b1_local is not None else None
    b2_t = b2_local.detach().clone().requires_grad_(True) if b2_local is not None else None

    y_local, _expert_freq = moe_ep_general_routing_forward(
        x_t,
        idx_local,
        scores_t,
        w1_t,
        b1_t,
        w2_t,
        b2_t,
        E=E,
        group=group,
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=False,
        concat_layout=concat_layout,
        ep_config=cfg,
        CPU_sync_on_runtime=CPU_sync_on_runtime,
    )
    inputs = [x_t, scores_t, w1_t, w2_t]
    if b1_t is not None:
        inputs += [b1_t, b2_t]
    grads = torch.autograd.grad(y_local, inputs, grad_outputs=dout_local, retain_graph=False)

    dx_local, ds_local, dw1_local, dw2_local = grads[:4]
    db1_local = grads[4] if b1_t is not None else None
    db2_local = grads[5] if b1_t is not None else None

    y_full = _all_gather_y(y_local.detach(), world_size, group)
    dx_full = _all_gather_y(dx_local, world_size, group)
    # ds is per-token like dx: the backward reduce-scatters it, so each rank's (T_local, K)
    # slice is already complete and needs no cross-rank sum (unlike TC's replicated router_w).
    ds_full = _all_gather_y(ds_local, world_size, group)
    dw1_list = _gather_to_rank0(dw1_local, world_size, rank, group)
    dw2_list = _gather_to_rank0(dw2_local, world_size, rank, group)
    db1_list = _gather_to_rank0(db1_local, world_size, rank, group) if db1_local is not None else None
    db2_list = _gather_to_rank0(db2_local, world_size, rank, group) if db2_local is not None else None

    if rank == 0:
        ep_dw1 = torch.cat(dw1_list, dim=2)  # (2I, H, E_local) per rank -> (2I, H, E)
        ep_dw2 = torch.cat(dw2_list, dim=0)  # (E_local, I, H) per rank -> (E, I, H)
        ep_db1 = torch.cat(db1_list, dim=0) if db1_list is not None else None
        ep_db2 = torch.cat(db2_list, dim=0) if db2_list is not None else None
        return y_full, dx_full, ds_full, ep_dw1, ep_dw2, ep_db1, ep_db2
    return None


# ============================================================================
# Per-shape test driver
# ============================================================================


@dataclass
class ShapeStats:
    shape_name: str
    pass_count: int = 0
    fail_count: int = 0
    failures: list = field(default_factory=list)


def _ep_topk_indices_global(
    x_local: torch.Tensor,
    router_w: torch.Tensor,
    K: int,
    W_E_local: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
    world_size: int,
    T_local: int,
    T: int,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Run the TC topk on each rank's local logits, all-gather (scoped to `group`,
    None = the default WORLD group), return the flat (T, K) int64 indices. Used to
    seed the reference so it picks the same experts that EP's TC topk did
    (deterministic tie-breaking match)."""
    with torch.no_grad():
        logits = F.linear(x_local, router_w)
        _, topk_idx_local = TC_Softmax_Topk_Router_Function.apply(
            logits,
            W_E_local,
            K,
            is_softmax_over_topk,
            norm_topk_probs,
        )
    topk_idx_global = torch.empty(world_size, T_local, K, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(
        topk_idx_global.view(-1),
        topk_idx_local.view(-1).contiguous(),
        group=group,
    )
    return topk_idx_global.view(T, K).to(torch.int64)


def _run_one_shape(
    rank: int,
    world_size: int,
    device: torch.device,
    shape: Shape,
    dtype: torch.dtype,
    concat_layout: bool,
    atol: float,
    rtol: float,
    seed: int,
    general_preflight: bool = False,
    only_nccl: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    group_id: Optional[int] = None,
    cpu_sync_on_runtime: bool = False,
) -> ShapeStats:
    # All collectives below are scoped to `group` (rank/world_size are already
    # group-relative) so this same driver also serves an independent EP group
    # among several siblings under --ep-world-size; None means the default WORLD
    # group, matching every pre-existing single-group invocation unchanged.
    src0 = 0 if group is None else dist.get_global_rank(group, 0)
    mode_pairs = [(DispatchMode.A2A_NCCL, CombineMode.A2A_NCCL)] if only_nccl else DISPATCH_COMBINE_MODE_PAIRS
    T, H, I, E, K = shape.T, shape.H, shape.I, shape.E, shape.K
    assert T % world_size == 0, f"T ({T}) must be divisible by world_size ({world_size})."
    assert E % world_size == 0, f"E ({E}) must be divisible by world_size ({world_size})."

    T_local = T // world_size
    E_local = E // world_size
    e_slc = slice(rank * E_local, (rank + 1) * E_local)
    stats = ShapeStats(shape.name)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType.SWIGLU,
            add_bias=True,
            std=0.02,
        )
        .to(dtype=dtype)
        .to(device)
    )
    torch.nn.init.normal_(moe.c_fc.bias, 0, 0.01)
    torch.nn.init.normal_(moe.c_proj.bias, 0, 0.01)

    # Belt-and-suspenders broadcast in case of any non-deterministic init.
    for p in moe.parameters():
        dist.broadcast(p.data, src=src0, group=group)

    w1_full = moe.c_fc.weight  # (E, 2I, H)
    w2_full = moe.c_proj.weight  # (E, H, I)
    b1_with = moe.c_fc.bias  # (E, 2I)
    b2_with = moe.c_proj.bias  # (E, H)
    router_w = moe.router.weight  # (E, H)

    # EP-sharded weights (per-rank expert slice). Layout matches the benchmark
    # in benchmarks/distributed/moe-ep.py:
    #   w1: (E_local, 2I, H) → permute(1, 2, 0) → (2I, H, E_local) view,
    #       strides (H, 1, 2I·H). Non-contiguous on purpose; the GEMM kernel
    #       requires the middle dim to have stride 1.
    #   w2: (E_local, H, I)  → permute(0, 2, 1).contiguous() → (E_local, I, H)
    #       contig.
    w1_local = w1_full[e_slc].permute(1, 2, 0)
    w2_local = w2_full[e_slc].permute(0, 2, 1).contiguous()
    b1_local_with = b1_with[e_slc].contiguous()
    b2_local_with = b2_with[e_slc].contiguous()

    if rank == 0:
        x_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        x_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(x_global, src=src0, group=group)
    x_local = x_global[rank * T_local : (rank + 1) * T_local].contiguous()

    # dout used for the backward seed — broadcast so every rank has the
    # same view and the rank-0 reference matches what each EP rank computes.
    if rank == 0:
        dout_global = 0.2 * torch.randn(T, H, device=device, dtype=dtype)
    else:
        dout_global = torch.empty(T, H, device=device, dtype=dtype)
    dist.broadcast(dout_global, src=src0, group=group)
    dout_local = dout_global[rank * T_local : (rank + 1) * T_local].contiguous()

    # Pre-broadcast a fixed routing decision used by general_routing_forward.
    if rank == 0:
        with torch.no_grad():
            scores_g, idx_g = F.linear(x_global, router_w).topk(K, dim=-1)
            scores_g = scores_g.softmax(dim=-1, dtype=torch.float32).to(dtype)
            idx_g = idx_g.to(torch.int64)
    else:
        scores_g = torch.empty(T, K, device=device, dtype=dtype)
        idx_g = torch.empty(T, K, device=device, dtype=torch.int64)
    dist.broadcast(scores_g, src=src0, group=group)
    dist.broadcast(idx_g, src=src0, group=group)
    scores_local = scores_g[rank * T_local : (rank + 1) * T_local].contiguous()
    idx_local = idx_g[rank * T_local : (rank + 1) * T_local].to(torch.int32).contiguous()

    if general_preflight:
        cfg = RuntimeEPConfig(dispatch_mode=DISPATCH_MODES[0], W=world_size, K=K, combine_mode=COMBINE_MODES[0])
        tag = f"general-preflight[dispatch={DISPATCH_MODES[0].value},combine={COMBINE_MODES[0].value}]"
        try:
            _run_ep_general_one_train(
                x_local,
                idx_local,
                scores_local,
                w1_local,
                w2_local,
                None,
                None,
                dout_local,
                E,
                cfg,
                concat_layout,
                world_size,
                rank,
                group,
                CPU_sync_on_runtime=cpu_sync_on_runtime,
            )
            if rank == 0:
                print(f"{tag} PASS — training-mode general routing forward+backward succeeded, no ICE")
        except Exception:
            if rank == 0:
                print(f"{tag} FAIL — exception during training-mode general routing:")
                traceback.print_exc()
            raise
        return stats

    # ------------------------------------------------------------------------
    # Sweep: bias × routing-variant × 3×3 (dispatch × combine) × entry point.
    # ------------------------------------------------------------------------
    for use_bias in (False, True):
        b1_local = b1_local_with if use_bias else None
        b2_local = b2_local_with if use_bias else None
        b1_full = b1_with if use_bias else None
        b2_full = b2_with if use_bias else None
        grp_tag = f"grp={group_id} " if group_id is not None else ""
        log_prefix = f"[{grp_tag}W={world_size} {shape.name} bias={int(use_bias)}] "

        # ────────── entry point #1: TC_softmax_topk_forward ──────────
        for variant_name, is_softmax_over_topk, norm_topk_probs in ROUTING_VARIANTS:
            # Use TC's own topk decision (gathered globally) so the
            # reference picks the same experts under ties.
            topk_idx_global = _ep_topk_indices_global(
                x_local,
                router_w,
                K,
                world_size * E_local,
                is_softmax_over_topk,
                norm_topk_probs,
                world_size,
                T_local,
                T,
                device,
                group,
            )
            if rank == 0:
                ref = _per_expert_reference_tc(
                    x_global,
                    router_w,
                    w1_full,
                    w2_full,
                    b1_full,
                    b2_full,
                    topk_idx_global,
                    is_softmax_over_topk,
                    norm_topk_probs,
                    concat_layout,
                    dout_global,
                )
                ref_o, ref_dx, ref_drouter_w, ref_dw1, ref_dw2, ref_db1, ref_db2 = ref

            for dispatch_mode, combine_mode in mode_pairs:
                cfg = RuntimeEPConfig(
                    dispatch_mode=dispatch_mode,
                    W=world_size,
                    K=K,
                    combine_mode=combine_mode,
                )
                tag = f"TC[{variant_name},dispatch={dispatch_mode.value},combine={combine_mode.value}]"
                try:
                    result = _run_ep_tc_one(
                        x_local,
                        router_w,
                        w1_local,
                        w2_local,
                        b1_local,
                        b2_local,
                        dout_local,
                        K,
                        E,
                        cfg,
                        is_softmax_over_topk,
                        norm_topk_probs,
                        concat_layout,
                        world_size,
                        rank,
                        group,
                        CPU_sync_on_runtime=cpu_sync_on_runtime,
                    )
                except Exception as e:
                    if rank == 0:
                        print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                        stats.fail_count += 1
                        stats.failures.append(f"{tag} (exception)")
                    dist.barrier()
                    continue
                if rank == 0:
                    y_full, ep_dx, ep_drouter_w, ep_dw1, ep_dw2, ep_db1, ep_db2 = result
                    # backward returns expert grads pre-divided by W (see ep.py); the oracle is
                    # a global sum, so scale it. drouter_w is exempt: already SUM-reduced.
                    quantities = [
                        ("o", y_full, ref_o),
                        ("dx", ep_dx, ref_dx),
                        ("drouter_w", ep_drouter_w, ref_drouter_w),
                        ("dw1", ep_dw1, ref_dw1.permute(1, 2, 0) / world_size),  # (2I, H, E)
                        ("dw2", ep_dw2, ref_dw2.permute(0, 2, 1) / world_size),  # (E, I, H)
                    ]
                    if ep_db1 is not None:
                        quantities.append(("db1", ep_db1, ref_db1 / world_size))
                        quantities.append(("db2", ep_db2, ref_db2 / world_size))
                    ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                    print(msg)
                    if ok:
                        stats.pass_count += 1
                    else:
                        stats.fail_count += 1
                        stats.failures.append(tag)
                dist.barrier()

        # ────────── entry point #2: general_routing_forward ──────────
        if rank == 0:
            ref_o_g, ref_dx_g, ref_ds_g, ref_dw1_g, ref_dw2_g, ref_db1_g, ref_db2_g = (
                _per_expert_reference_general(
                    x_global,
                    w1_full,
                    w2_full,
                    b1_full,
                    b2_full,
                    idx_g,
                    scores_g,
                    concat_layout,
                    dout_global,
                )
            )

        for dispatch_mode, combine_mode in mode_pairs:
            cfg = RuntimeEPConfig(
                dispatch_mode=dispatch_mode,
                W=world_size,
                K=K,
                combine_mode=combine_mode,
            )
            tag = f"general[dispatch={dispatch_mode.value},combine={combine_mode.value}]"
            try:
                y_full = _run_ep_general_one_fwd(
                    x_local,
                    idx_local,
                    scores_local,
                    w1_local,
                    w2_local,
                    b1_local,
                    b2_local,
                    E,
                    cfg,
                    concat_layout,
                    world_size,
                    rank,
                    group,
                    CPU_sync_on_runtime=cpu_sync_on_runtime,
                )
            except Exception as e:
                if rank == 0:
                    print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                    stats.fail_count += 1
                    stats.failures.append(f"{tag} (exception)")
                dist.barrier()
                continue
            if rank == 0:
                quantities = [("o", y_full, ref_o_g)]
                ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()

        # ────────── entry point #2b: general_routing_forward in training mode (fwd + bwd) ──────────
        for dispatch_mode, combine_mode in mode_pairs:
            cfg = RuntimeEPConfig(
                dispatch_mode=dispatch_mode,
                W=world_size,
                K=K,
                combine_mode=combine_mode,
            )
            tag = f"general-train[dispatch={dispatch_mode.value},combine={combine_mode.value}]"
            try:
                result = _run_ep_general_one_train(
                    x_local,
                    idx_local,
                    scores_local,
                    w1_local,
                    w2_local,
                    b1_local,
                    b2_local,
                    dout_local,
                    E,
                    cfg,
                    concat_layout,
                    world_size,
                    rank,
                    group,
                    CPU_sync_on_runtime=cpu_sync_on_runtime,
                )
            except Exception as e:
                if rank == 0:
                    print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                    stats.fail_count += 1
                    stats.failures.append(f"{tag} (exception)")
                dist.barrier()
                continue
            if rank == 0:
                y_full, ep_dx, ep_ds, ep_dw1, ep_dw2, ep_db1, ep_db2 = result
                # expert grads come back pre-divided by W (see ep.py); the oracle is a global
                # sum, so scale it. o/dx/ds are per-token quantities and need no scaling.
                quantities = [
                    ("o", y_full, ref_o_g),
                    ("dx", ep_dx, ref_dx_g),
                    ("ds", ep_ds, ref_ds_g),
                    ("dw1", ep_dw1, ref_dw1_g.permute(1, 2, 0) / world_size),  # (2I, H, E)
                    ("dw2", ep_dw2, ref_dw2_g.permute(0, 2, 1) / world_size),  # (E, I, H)
                ]
                if ep_db1 is not None:
                    quantities.append(("db1", ep_db1, ref_db1_g / world_size))
                    quantities.append(("db2", ep_db2, ref_db2_g / world_size))
                ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()

        # ────────── entry point #2c: general-train with dropped (-1) slots, A2A_NCCL only ──────────
        nccl_cfg = RuntimeEPConfig(
            dispatch_mode=DispatchMode.A2A_NCCL, W=world_size, K=K, combine_mode=CombineMode.A2A_NCCL
        )
        drop_routings = [
            ("rand10", partial(_random_drop_routing, idx_g, seed)),
            ("hot-cap1", partial(_hot_capped_routing, scores_g, T_local, E, world_size)),
        ]
        for drop_name, build_routing in drop_routings:
            tag = f"general-train-drop[{drop_name},dispatch=a2a_nccl,combine=a2a_nccl]"
            try:
                idx_d = build_routing()
                result = _run_ep_general_one_train(
                    x_local,
                    idx_d[rank * T_local : (rank + 1) * T_local].to(torch.int32).contiguous(),
                    scores_local,
                    w1_local,
                    w2_local,
                    b1_local,
                    b2_local,
                    dout_local,
                    E,
                    nccl_cfg,
                    concat_layout,
                    world_size,
                    rank,
                    group,
                    CPU_sync_on_runtime=cpu_sync_on_runtime,
                )
            except Exception as e:
                if rank == 0:
                    print(f"{log_prefix}{tag:<88s} ✗ EXC   {type(e).__name__}: {str(e)[:160]}")
                    stats.fail_count += 1
                    stats.failures.append(f"{tag} (exception)")
                dist.barrier()
                continue
            if rank == 0:
                # the oracle's per-expert match never selects -1, so dropped slots add nothing and get ds = 0
                ref_o_d, ref_dx_d, ref_ds_d, ref_dw1_d, ref_dw2_d, ref_db1_d, ref_db2_d = (
                    _per_expert_reference_general(
                        x_global, w1_full, w2_full, b1_full, b2_full, idx_d, scores_g, concat_layout, dout_global
                    )
                )
                y_full, ep_dx, ep_ds, ep_dw1, ep_dw2, ep_db1, ep_db2 = result
                quantities = [
                    ("o", y_full, ref_o_d),
                    ("dx", ep_dx, ref_dx_d),
                    ("ds", ep_ds, ref_ds_d),
                    ("dw1", ep_dw1, ref_dw1_d.permute(1, 2, 0) / world_size),
                    ("dw2", ep_dw2, ref_dw2_d.permute(0, 2, 1) / world_size),
                ]
                if ep_db1 is not None:
                    quantities.append(("db1", ep_db1, ref_db1_d / world_size))
                    quantities.append(("db2", ep_db2, ref_db2_d / world_size))
                ok, msg = _check_quantities(tag, log_prefix, quantities, atol, rtol)
                print(msg)
                if ok:
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1
                    stats.failures.append(tag)
            dist.barrier()
        dist.barrier()

    return stats


def _random_drop_routing(idx_g: torch.Tensor, seed: int) -> torch.Tensor:
    """~10% of slots set to -1; CPU-seeded so every rank builds the same routing without a broadcast."""
    drop = torch.rand(idx_g.shape, generator=torch.Generator().manual_seed(seed)) < 0.1
    return torch.where(drop.to(idx_g.device), -1, idx_g)


def _hot_capped_routing(scores_g: torch.Tensor, T_local: int, E: int, world_size: int) -> torch.Tensor:
    """Every token routes to experts 0..K-1 (all on the lowest ranks), then each source rank keeps its
    highest-scored ceil(T_local*K/W) slots per destination rank, so the truncation binds end to end."""
    T, K = scores_g.shape
    E_local = E // world_size
    capacity = -(-T_local * K // world_size)
    capped = torch.arange(K, device=scores_g.device).expand(T, K).contiguous()
    dst = capped // E_local
    for r in range(world_size):
        rows = slice(r * T_local, (r + 1) * T_local)
        flat_dst, flat_scores, flat_out = dst[rows].flatten(), scores_g[rows].flatten(), capped[rows].view(-1)
        for d in range(world_size):
            slots = (flat_dst == d).nonzero().flatten()
            order = flat_scores[slots].argsort(descending=True, stable=True)
            flat_out[slots[order[capacity:]]] = -1
    assert (capped < 0).any(), "hot routing should exceed the capacity cap"
    return capped


def _print_summary(all_stats: List[ShapeStats], group_id: Optional[int] = None) -> bool:
    total_pass = sum(s.pass_count for s in all_stats)
    total_fail = sum(s.fail_count for s in all_stats)
    grp_tag = f" (grp={group_id})" if group_id is not None else ""
    print(f"\n=== Summary{grp_tag} ===")
    name_w = max((len(s.shape_name) for s in all_stats), default=10)
    for s in all_stats:
        marker = "✓" if s.fail_count == 0 else "✗"
        print(f"  {marker} {s.shape_name:<{name_w}}  pass={s.pass_count}  fail={s.fail_count}")
        for f in s.failures:
            print(f"      - {f}")
    print(f"\nTotal: pass={total_pass}  fail={total_fail}")
    return total_fail == 0


def _under_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _check_row_bins() -> bool:
    """Pure-Python sanity check for ep._row_bins's grid on a known shape: no
    GPU/dist needed, so this always runs, even without torchrun."""
    max_rows = 8192 * 4 * 4  # T_local=8192, K=4, W=4, E_local=12 -> x = T_local*K = 32768
    expected = (16384, 24576, 32768, 40960, 49152, 65536, 98304, 131072)
    actual = _row_bins(max_rows, W=4, K=4, E_local=12)
    ok = actual == expected
    if not ok:
        print(f"_row_bins MISMATCH: expected {expected}, got {actual}", file=sys.stderr)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concat-layout",
        action="store_true",
        help="Test the concat [g; u] up-proj layout instead of interleaved.",
    )
    parser.add_argument(
        "--general-preflight",
        action="store_true",
        help="ep-port-phase2 step 1 pre-flight: run one general-routing fwd+bwd in "
        "training mode (K_eq_4 shape, one dispatch/combine pair) and report whether "
        "the QuACK DSL ICE still reproduces. Skips the full sweep.",
    )
    parser.add_argument(
        "--only-nccl",
        action="store_true",
        help="Restrict the sweep to (A2A_NCCL, A2A_NCCL) only, skipping every symm-mem "
        "mode pair. Use this for genuine multi-node runs: AG/A2A/RANK_DEDUP dispatch and "
        "RS/RANK_DEDUP combine are all symm-mem (NVLink P2P) based and are NOT expected to "
        "work across nodes -- whether they hang or throw a CUDA driver error (observed: "
        "'invalid resource handle') depends on undecidable-from-here physical NVLink "
        "topology between whichever nodes SLURM happened to allocate, not on any bug. "
        "Single-node --standalone runs should omit this flag to keep full sweep coverage.",
    )
    parser.add_argument(
        "--ep-world-size",
        type=int,
        default=None,
        help="Split WORLD_SIZE ranks into WORLD_SIZE/ep-world-size independent EP groups "
        "of this many ranks each (contiguous rank blocks), to exercise ep < dps "
        "(fsdp_ep > 1) hybrid topologies. Defaults to WORLD_SIZE (a single group, the "
        "existing behavior). Implies --only-nccl regardless of that flag's own value, "
        "since a hybrid group's ranks are placed to be cross-node by construction.",
    )
    parser.add_argument(
        "--cpu-sync-on-runtime",
        action="store_true",
        help="Pass CPU_sync_on_runtime=True to both entry points, exercising the symm "
        "path's binned max_rows_per_rank_runtime branch (lm-engine's production setting; "
        "otherwise untested here). The NCCL path always bins regardless of this flag.",
    )
    args = parser.parse_args()

    if not _check_row_bins():
        return 2

    if not _under_torchrun():
        print(
            "ERROR: this test must be launched with torchrun, e.g.:\n"
            "  torchrun --nproc_per_node=8 --standalone tests/moe_ep_test.py",
            file=sys.stderr,
        )
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size < 2:
        if rank == 0:
            print(f"SKIP: EP test needs world_size >= 2 (got {world_size}).")
        return 0
    ep_world_size = args.ep_world_size if args.ep_world_size is not None else world_size
    assert world_size % ep_world_size == 0, (
        f"WORLD_SIZE ({world_size}) must be a multiple of --ep-world-size ({ep_world_size})"
    )
    only_nccl = args.only_nccl or ep_world_size < world_size
    if local_rank >= torch.cuda.device_count():
        print(
            f"[r{rank}] ERROR: LOCAL_RANK={local_rank} but only " f"{torch.cuda.device_count()} CUDA devices visible",
            file=sys.stderr,
        )
        return 2

    # Match the benchmark's numerics: disable TF32 so the fp32 reference path
    # (F.linear in the references) is genuinely fp32 and not silently downcast
    # on Ampere+.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.set_device(local_rank)
    # Default NCCL timeout (~10 min) is too tight once W grows: Triton/QuACK
    # autotune over a new (T, H, I, E, K) shape can cold-compile for minutes,
    # and with more ranks a single straggler is more likely -- exactly the
    # class of issue bench-ep-comm.py's own 60-minute override documents.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=datetime.timedelta(minutes=60),
    )
    device = torch.device(f"cuda:{local_rank}")

    # Every rank creates every contiguous-block subgroup, in the same order, and keeps
    # only the one containing its own global rank -- new_group() requires uniform
    # participation across the whole world even though only one block's members end up
    # using each group. A single block (ep_world_size == world_size) reduces to one group
    # spanning WORLD, i.e. group_id is None and every _run_one_shape call behaves exactly
    # as before this flag existed.
    num_groups = world_size // ep_world_size
    if num_groups > 1:
        groups = [dist.new_group(ranks=list(range(g * ep_world_size, (g + 1) * ep_world_size))) for g in range(num_groups)]
        group_id = rank // ep_world_size
        ep_group = groups[group_id]
        ep_rank = rank - group_id * ep_world_size
    else:
        group_id = None
        ep_group = None
        ep_rank = rank

    if rank == 0:
        print(
            f"\nEP correctness test (W={world_size}, ep_world_size={ep_world_size}, "
            f"num_ep_groups={num_groups}, "
            f"concat_layout={args.concat_layout}, "
            f"shapes={len(SHAPES)})\n"
            f"per (bias × routing) cell exercises 3 dispatch × 3 combine modes\n"
            f"for both TC_softmax_topk and general_routing entry points,\n"
            f"comparing fwd + bwd grads against an fp32 PyTorch autograd reference.\n"
        )

    shapes = [s for s in SHAPES if s.name == "K_eq_4"] if args.general_preflight else SHAPES
    all_stats: List[ShapeStats] = []
    try:
        for shape in shapes:
            try:
                stats = _run_one_shape(
                    ep_rank,
                    ep_world_size,
                    device,
                    shape,
                    dtype=torch.bfloat16,
                    concat_layout=args.concat_layout,
                    atol=5e-2,
                    rtol=5e-2,
                    seed=1111,
                    general_preflight=args.general_preflight,
                    only_nccl=only_nccl,
                    group=ep_group,
                    group_id=group_id,
                    cpu_sync_on_runtime=args.cpu_sync_on_runtime,
                )
            except Exception as e:
                if ep_rank == 0:
                    print(f"[ERR {shape.name}] {e}")
                    traceback.print_exc()
                stats = ShapeStats(shape.name)
                stats.fail_count = 1
                stats.failures.append(f"exception: {e}")
            all_stats.append(stats)
            torch.cuda.empty_cache()
    finally:
        # Each EP group's local rank 0 ("leader") decides its own group's pass/fail (with
        # num_groups==1 -- the pre-existing single-group case -- there is exactly one
        # leader, global rank 0, so this reduces to that case exactly). Every rank
        # contributes a neutral 1 except leaders, who contribute their real verdict; a
        # world-wide MIN then ANDs every group's result together and delivers it to every
        # rank, replacing the old rank-0-decides-then-broadcast pattern (which only knew
        # about a single group) with one that generalizes to any number of them.
        local_ok = 1
        if ep_rank == 0:
            local_ok = 1 if _print_summary(all_stats, group_id) else 0
        success_t = torch.tensor([local_ok], device=device, dtype=torch.int32)
        dist.all_reduce(success_t, op=dist.ReduceOp.MIN)
        success = bool(success_t.item())

        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

    return 0 if success else 1


if __name__ == "__main__":
    # Hard-exit via os._exit to bypass the Python destructor chain on
    # paths where ``clear_ep_cache``'s atexit hook may not get to run
    # ahead of ``~CUDASymmetricMemory → cuMemUnmap`` (e.g. a test
    # exception that propagates past atexit ordering). Same pattern as
    # ``benchmarks/distributed/moe-ep.py`` and ``tests/distributed/
    # collectives_test.py``.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)

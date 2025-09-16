# sanity_matrix.py
import json
import sys
import pathlib
import numpy as np
import torch
import yaml

from evo_search.architectures.hat_arch import hat_architecture
from fairseq.modules import LinearSuper
from fairseq.modules.multihead_attention_super import MultiheadAttentionSuper


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _set_selection_modes(problem, *, lin_mode="prefix", mha_mode="prefix", metric="l2"):
    """
    lin_mode: 'prefix' or 'topk'
    mha_mode: 'prefix' or 'magnitude'
    metric:   'l1' or 'l2'
    """
    model = problem.model
    for m in model.modules():
        if isinstance(m, LinearSuper):
            m.selection_mode = 'topk' if lin_mode == 'topk' else 'prefix'
            m.metric = metric

            # If your LinearSuper has column/row overrides, make sure we don't
            # carry a stale override across modes. If None is not accepted,
            # fall back to "full range".
            if hasattr(m, "set_col_idx_override"):
                try:
                    m.set_col_idx_override(None)  # works if your method accepts None
                except Exception:
                    # fallback to full range (acts like prefix)
                    full_cols = torch.arange(m.super_in_dim, device=m.weight.device, dtype=torch.long)
                    m.set_col_idx_override(full_cols)

            if hasattr(m, "set_row_idx_override"):
                try:
                    m.set_row_idx_override(None)
                except Exception:
                    # full range rows (acts like prefix)
                    full_rows = torch.arange(m.super_out_dim, device=m.weight.device, dtype=torch.long)
                    m.set_row_idx_override(full_rows)

        if isinstance(m, MultiheadAttentionSuper):
            # our modified MHA supports 'prefix' and 'magnitude'
            m.selection_mode = 'magnitude' if mha_mode == 'magnitude' else 'prefix'
            m.metric = metric
            # clear any previously kept heads to force recompute on next set_sample_config
            if hasattr(m, "_keep_heads"):
                m._keep_heads = None
            # and ensure out_proj is not stuck with a stale override from a previous run
            if hasattr(m, "out_proj") and hasattr(m.out_proj, "set_col_idx_override"):
                try:
                    m.out_proj.set_col_idx_override(None)
                except Exception:
                    # fallback: set to full qkv range (acts like prefix)
                    # this is safe because out_proj.super_in_dim == qkv_dim in HAT
                    full_cols = torch.arange(m.qkv_dim, device=m.out_proj.weight.device, dtype=torch.long)
                    m.out_proj.set_col_idx_override(full_cols)

    model.eval()
    torch.set_grad_enabled(False)


def _quick_eval(problem, genome, batches=10):
    """
    Use your existing pipeline to evaluate loss on `batches` from valid set.
    """
    genome = np.array(genome)
    return problem.evaluate_validation_loss(genome, max_batches=batches)


# --- main sanity driver ------------------------------------------------------

def main():
    # paths relative to your repo (same as your main.py)
    root = pathlib.Path(__file__).resolve().parent
    design_space_yml = root / "configs" / "hat_design_space.yaml"

    # 1) Build the problem (loads model, task, criterion, dataset, etc.)
    design_space = _load_yaml(design_space_yml)
    problem = hat_architecture(design_space)

    # 2) Fixed genome to compare apples-to-apples across modes
    g = np.array([
        6, 2, 0, 1,
        0, 1, 1, 1, 2, 1,
        0, 0, -1, -1, -1, -1,
        1, 0, 1, 1, 0,
        1, 1, 0,
        -1, -1, -1, -1,
        0, 1,
        -1, -1, -1, -1,
        0, 0, -1, -1, -1, -1
    ])

    # How many validation batches to use for the quick sanity loss
    batches = 10
    if len(sys.argv) > 1:
        try:
            batches = int(sys.argv[1])
        except Exception:
            pass

    print("\n=== SANITY MATRIX (same genome, different modes) ===")
    print("Genome:", g.tolist(), "\n")

    # A) lin=prefix,  mha=prefix
    _set_selection_modes(problem, lin_mode="prefix", mha_mode="prefix", metric="l2")
    loss_pp = _quick_eval(problem, g, batches=batches)
    print(f"lin=prefix,    mha=prefix     -> loss={loss_pp:.4f}")

    # B) lin=topk,    mha=prefix
    _set_selection_modes(problem, lin_mode="topk", mha_mode="prefix", metric="l2")
    loss_tp = _quick_eval(problem, g, batches=batches)
    print(f"lin=topk,      mha=prefix     -> loss={loss_tp:.4f}")

    # C) lin=prefix,  mha=magnitude
    _set_selection_modes(problem, lin_mode="prefix", mha_mode="magnitude", metric="l2")
    loss_pm = _quick_eval(problem, g, batches=batches)
    print(f"lin=prefix,    mha=magnitude  -> loss={loss_pm:.4f}")

    # D) lin=topk,    mha=magnitude
    _set_selection_modes(problem, lin_mode="topk", mha_mode="magnitude", metric="l2")
    loss_tm = _quick_eval(problem, g, batches=batches)
    print(f"lin=topk,      mha=magnitude  -> loss={loss_tm:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
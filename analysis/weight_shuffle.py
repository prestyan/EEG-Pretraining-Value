"""Distribution-preserving perturbations of pretrained projection parameters.

Call after loading a fresh pretrained model and before creating the optimizer.
The input module is modified in place; the return value is a JSON-safe report.
"""

import hashlib
import math
from numbers import Integral, Real

import torch
from torch import nn


_PROJECTION_TYPES = (
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)
_ATTENTION_WEIGHTS = {
    "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
}


def _generator(seed, name, purpose):
    # Separate streams keep selection independent of the shuffle ratio and of
    # other layers. CPU indices make the perturbation independent of device.
    digest = hashlib.sha256(f"{seed}:{name}:{purpose}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63)
    return torch.Generator(device="cpu").manual_seed(local_seed)


def _projection_parameters(backbone):
    """Yield stored projection weights, including weight-norm direction v."""
    seen = set()
    for module_name, module in backbone.named_modules():
        candidates = []
        if isinstance(module, _PROJECTION_TYPES):
            # Legacy weight_norm stores v and g instead of weight. Only v is
            # shuffled; the scale g is preserved.
            if isinstance(getattr(module, "weight_v", None), nn.Parameter):
                candidates.append(("weight_v", module.weight_v))
            elif hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                parametrization = module.parametrizations.weight
                if (len(parametrization) == 1
                        and type(parametrization[0]).__name__ == "_WeightNorm"
                        and hasattr(parametrization, "original1")):
                    candidates.append(("parametrizations.weight.original1", parametrization.original1))
                else:
                    raise ValueError(
                        f"Unsupported weight parametrization in {module_name!r}; "
                        "only ordinary weights and weight_norm are supported."
                    )
            elif isinstance(getattr(module, "weight", None), nn.Parameter):
                candidates.append(("weight", module.weight))
        elif isinstance(module, nn.MultiheadAttention):
            candidates.extend(
                (name, parameter)
                for name, parameter in module.named_parameters(recurse=False)
                if name in _ATTENTION_WEIGHTS
            )

        for local_name, parameter in candidates:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim >= 2 and parameter.is_floating_point():
                name = f"{module_name}.{local_name}" if module_name else local_name
                yield name, parameter


@torch.no_grad()
def shuffle_pretrained_weights(model_choice, model, ratio=0.25, seed=2025, *, verbose=True):
    """Shuffle a fraction of the entries inside each backbone projection tensor.

    Parameters
    ----------
    model_choice : str
        Experiment label, e.g. EEGPT / CBRAMOD / LABRAM / CODEBRAIN / BIOT /
        BENDR / SINGLEM. There is no checkpoint-format-specific processing.
    model : torch.nn.Module
        A loaded backbone, or your LitModelCausal instance. If ``.backbone``
        exists, only that submodule is touched. Wrappers explicitly declaring
        ``is_load_pretrained=False`` are left unchanged (scratch control).
    ratio : float
        In [0, 1]. Select floor(ratio * numel) entries independently per eligible
        tensor, then randomly permute their existing values. Unselected entries
        remain identical. Zero is an exact no-op. Some selected entries may keep
        their value, so selected_fraction and changed_fraction can differ.
    seed : int
        Local shuffle seed; does not advance the global training RNG. Use the
        same seed and original checkpoint for paired frozen/fine-tuned runs.

    Notes
    -----
    Modifies Linear/Conv projection weights and MultiheadAttention projections.
    Skips biases, normalization, embeddings, tokens, buffers and custom SSM
    parameters. This also works with requires_grad=False. Device, dtype,
    Parameter identity, training mode and requires_grad are preserved.

    For weight_norm, the stored direction parameter v is shuffled, with g fixed.
    Its value distribution is preserved, not necessarily the distribution of
    the derived effective weight g*v/||v||. The report lists the exact scope.

    Always reload the original checkpoint before another ratio/seed. Repeated
    calls on the same instance compose perturbations. A ratio of 1 means all
    eligible projection entries, not all encoder parameters or scratch init.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module with weights already loaded")
    if not isinstance(model_choice, str) or not model_choice.strip():
        raise ValueError("model_choice must be a nonempty string")
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be a finite number in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    choice = model_choice.strip().upper()
    if hasattr(model, "model_choice") and str(model.model_choice).upper() != choice:
        raise ValueError("model_choice does not match model.model_choice")
    backbone = getattr(model, "backbone", model)
    if not isinstance(backbone, nn.Module):
        raise TypeError("model.backbone must be a torch.nn.Module")

    report = {
        "model_choice": choice,
        "ratio": float(ratio),
        "seed": int(seed),
        "scope": "linear_conv_attention_projection_parameters",
        "applied": False,
        "skipped_reason": None,
        "eligible_numel": 0,
        "selected_numel": 0,
        "changed_numel": 0,
        "selected_fraction": 0.0,
        "changed_fraction": 0.0,
        "tensors": [],
    }
    if getattr(model, "is_load_pretrained", True) is False:
        report["skipped_reason"] = "scratch_control"
        if verbose:
            print(f"[WeightShuffle][{choice}] skipped: scratch control")
        return report

    # Resolve/validate all candidate tensors before starting any mutation.
    candidates = list(_projection_parameters(backbone))
    if not candidates:
        raise ValueError("No supported real-valued projection weights found in backbone")
    for name, parameter in candidates:
        if parameter.device.type == "meta" or parameter.layout != torch.strided:
            raise ValueError(f"{name}: a materialized, strided weight is required")

    for name, parameter in candidates:
        n = parameter.numel()
        k = min(n, math.floor(float(ratio) * n))
        changed = 0
        if k >= 2:
            # A clone also handles noncontiguous Parameters without rebinding
            # their storage or losing optimizer/weight-sharing references.
            flat = parameter.detach().cpu().reshape(-1).clone()
            indices = torch.randperm(n, generator=_generator(seed, name, "select"))[:k]
            values = flat[indices]
            order = torch.randperm(k, generator=_generator(seed, name, "permute"))
            shuffled = values[order]
            changed = int(torch.count_nonzero(values != shuffled).item())
            flat[indices] = shuffled
            parameter.copy_(flat.reshape(parameter.shape).to(parameter.device))

        report["tensors"].append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": n,
            "selected_numel": k,
            "changed_numel": changed,
        })
        report["eligible_numel"] += n
        report["selected_numel"] += k
        report["changed_numel"] += changed

    total = report["eligible_numel"]
    report["selected_fraction"] = report["selected_numel"] / total if total else 0.0
    report["changed_fraction"] = report["changed_numel"] / total if total else 0.0
    report["applied"] = report["changed_numel"] > 0
    if verbose:
        print(
            f"[WeightShuffle][{choice}] ratio={ratio:g}, seed={seed}, "
            f"tensors={len(candidates)}, selected={report['selected_fraction']:.2%}, "
            f"actually_changed={report['changed_fraction']:.2%}"
        )
    return report

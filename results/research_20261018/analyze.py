"""CPU-only analysis of Dory's byte limits and LM-head screening feasibility.

Run with the repository's Python environment. Synthetic head inputs measure
candidate counts, not model accuracy or GPU latency. No CUDA work is submitted.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def limits(config):
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    dim = config["head_dim"]
    intermediate = config["intermediate_size"]
    pair_bytes = ((3 * heads + 2 * kv_heads) * dim * hidden
                  + 3 * intermediate * hidden) * (0.5 + 1 / 16)
    recurrent_bytes = config["n_recurrent_layers"] / 2 * pair_bytes
    fixed_bytes = (config["n_input_layers"] + config["n_output_layers"]) / 2 * pair_bytes
    head_bf16 = config["vocab_size"] * hidden * 2
    head_int8 = config["vocab_size"] * (hidden + 4)
    peak = 1008e9
    rows = []
    for loops in (1, 2, 4):
        windows = []
        for index, layer_type in enumerate(config["layer_types"]):
            if layer_type == "mlp":
                continue
            repeats = loops if (config["n_input_layers"] <= index
                               < config["n_input_layers"] + config["n_recurrent_layers"]) else 1
            windows.extend([config["sliding_window"]
                            if layer_type == "sliding_attention" else None] * repeats)
        for prompt in (640, 8192, 32768, 65536):
            # Match gap_analysis.py's mean context over its 63 analyzed steps.
            kv_bytes = sum(min(prompt + 32, w) if w else prompt + 32 for w in windows)
            kv_bytes *= 2 * kv_heads * dim * 2
            weights = fixed_bytes + loops * recurrent_bytes
            row = dict(loops=loops, prompt_tokens=prompt,
                       full_attention_applications=windows.count(None),
                       sliding_attention_applications=len(windows) - windows.count(None),
                       weight_bytes=weights, kv_bytes=kv_bytes)
            for name, head in (("bf16_head", head_bf16), ("int8_head", head_int8)):
                ms = (weights + head + kv_bytes) / peak * 1000 + 0.01
                row[name] = dict(ideal_ms=ms, ideal_tokens_per_second=1000 / ms)
            # This intentionally optimistic bound omits extra draft-coda KV
            # reads, rejected work, synchronization and arithmetic overhead.
            wave_ms = (fixed_bytes + recurrent_bytes + head_bf16 + kv_bytes) / peak * 1000 + 0.01
            row["perfect_wavefront_optimistic_bf16_head"] = dict(
                ideal_ms=wave_ms, ideal_tokens_per_second=1000 / wave_ms)
            rows.append(row)
    return dict(peak_bytes_per_second=peak, pair_weight_bytes=pair_bytes,
                recurrent_block_weight_bytes=recurrent_bytes,
                fixed_block_weight_bytes=fixed_bytes, head_bf16_bytes=head_bf16,
                head_int8_bytes_including_row_scales=head_int8,
                caveats=["Analytical byte model, not measured performance.",
                         "Main matrix bytes and group scales; small row metadata omitted.",
                         "Wavefront assumes perfect acceptance and batched draft/target coda reads."],
                cases=rows)


def head_screening(config, samples):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    generator = torch.Generator(device="cpu").manual_seed(20261018)
    hidden = torch.randn(samples, config["hidden_size"], generator=generator)
    hidden = (hidden / hidden.norm(dim=1, keepdim=True)).to(torch.bfloat16).float()
    norms = hidden.norm(dim=1)
    approximate, reference, residuals, row_norms = [], [], [], []
    with safe_open(ROOT / "models/nvfp4/model.safetensors", framework="pt", device="cpu") as checkpoint:
        weights = checkpoint.get_slice("lm_head.weight")
        factors = checkpoint.get_tensor("lm_head.logit_scale").float() / config["initializer_range"]
        for start in range(0, config["vocab_size"], 4096):
            rows = weights[start:start + 4096].float()
            factor = factors[start:start + 4096]
            scale = rows.abs().amax(dim=1).clamp(min=1e-12) / 127
            decoded = (rows / scale[:, None]).round().clamp(-127, 127) * scale[:, None]
            approximate.append((hidden @ decoded.T) * factor)
            reference.append((hidden @ rows.T).to(torch.bfloat16).float() * factor)
            residuals.append((rows - decoded).norm(dim=1) * factor.abs())
            row_norms.append(rows.norm(dim=1) * factor.abs())
    approximate = torch.cat(approximate, dim=1)
    reference = torch.cat(reference, dim=1)
    residuals = torch.cat(residuals)
    row_norms = torch.cat(row_norms)
    geometric_error = norms[:, None] * residuals[None, :]
    # Enclose the final BF16 dot-product rounding as well as quantization.
    # The last term is a conservative FP32 dot-product allowance for this
    # CPU pilot. A production certificate must match the GPU's reductions,
    # row-scale rounding, denormals, and selection/tie-breaking rules.
    unit_bf16 = 2 ** -8
    n = config["hidden_size"]
    gamma = (n * 2 ** -24) / (1 - n * 2 ** -24)
    error = geometric_error * (1 + unit_bf16) + unit_bf16 * approximate.abs()
    error += 4 * gamma * norms[:, None] * (row_norms + residuals)[None, :]
    lower = (approximate - error).amax(dim=1)
    candidates = approximate + error >= lower[:, None]
    best = reference.argmax(dim=1)
    retained = candidates[torch.arange(samples), best]
    assert retained.all(), "Synthetic reference winner escaped the proposed envelope"
    assert torch.all((reference - approximate).abs() <= error), "Observed error escaped envelope"
    counts = candidates.sum(dim=1)

    def quantiles(values):
        return {str(q): float(torch.quantile(values.float(), q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)}

    return dict(device="cpu", seed=20261018, samples=samples,
                input_distribution="Random unit-norm vectors, rounded to BF16; not Dory activations.",
                residual_l2_bound_quantiles=quantiles(residuals),
                scaled_row_l2_quantiles=quantiles(row_norms),
                candidate_count_quantiles=quantiles(counts),
                candidate_counts=counts.tolist(),
                reference_winners_retained=int(retained.sum()),
                all_observed_reference_logits_inside_envelopes=True,
                approximate_top1_matches=int((approximate.argmax(dim=1) == best).sum()),
                mean_head_byte_fraction=float((config["vocab_size"] * (config["hidden_size"] + 8)
                                               + counts.float().mean() * config["hidden_size"] * 2)
                                              / (config["vocab_size"] * config["hidden_size"] * 2)),
                caveats=["Not a production floating-point certificate.",
                         "No GPU kernel timing or model-quality evaluation.",
                         "Candidate counts must be repeated on real hidden states.",
                         "Greedy selection only; full log probabilities need the full normalizer."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-samples", type=int, default=0)
    args = parser.parse_args()
    if args.head_samples < 0:
        parser.error("--head-samples must be nonnegative")
    config = json.loads((ROOT / "models/nvfp4/config.json").read_text())
    out = OUTPUT / "limits.json"
    out.write_text(json.dumps(limits(config), indent=2) + "\n")
    print(out)
    if args.head_samples:
        result = head_screening(config, args.head_samples)
        out = OUTPUT / "head_screening.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(out)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

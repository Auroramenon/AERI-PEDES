"""Read-only diagnostics for the ground-aerial semantic-slot target."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_pid_mismatch_indices(pids):
    """Choose one deterministic, different-PID partner per sample."""
    if pids.ndim != 1:
        raise ValueError("pids must have shape [B]")
    if pids.numel() < 2:
        raise ValueError("at least two samples are required")

    indices = []
    batch_size = pids.numel()
    for row in range(batch_size):
        partner = None
        for offset in range(1, batch_size):
            candidate = (row + offset) % batch_size
            if pids[candidate].item() != pids[row].item():
                partner = candidate
                break
        if partner is None:
            raise ValueError("batch does not contain a different PID")
        indices.append(partner)

    result = torch.tensor(indices, device=pids.device, dtype=torch.long)
    if torch.any(pids[result] == pids):
        raise RuntimeError("failed to construct PID-mismatched pairs")
    return result


def off_diagonal_values(similarity):
    """Flatten the non-diagonal part of batched square matrices."""
    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [B, K, K]")
    if similarity.shape[1] != similarity.shape[2]:
        raise ValueError("similarity matrices must be square")

    size = similarity.shape[1]
    mask = ~torch.eye(
        size, device=similarity.device, dtype=torch.bool
    )
    return similarity[:, mask]


def pairwise_slot_cosine(slots):
    """Return cosine similarity between every pair of slots."""
    if slots.ndim != 3:
        raise ValueError("slots must have shape [B, K, D]")
    unit = F.normalize(slots.float(), p=2, dim=-1)
    return torch.einsum("bkd,bjd->bkj", unit, unit)


def slot_attention(tokens, slot_pool):
    """Reproduce SemanticSlotPool attention without changing the model."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, L, D]")

    tokens = tokens.float()
    queries = slot_pool.slot_queries.float().unsqueeze(0).expand(
        tokens.shape[0], -1, -1
    )
    logits = torch.einsum(
        "bkd,bld->bkl", queries, tokens
    ) * slot_pool.scale
    return F.softmax(logits, dim=-1)


def normalized_attention_entropy(attention):
    """Return per-sample, per-slot entropy in the range [0, 1]."""
    if attention.ndim != 3:
        raise ValueError("attention must have shape [B, K, L]")
    token_count = attention.shape[-1]
    if token_count <= 1:
        raise ValueError("attention needs at least two tokens")

    entropy = -(
        attention.clamp_min(1e-12)
        * attention.clamp_min(1e-12).log()
    ).sum(dim=-1)
    return entropy / math.log(token_count)


def describe(values):
    """Return JSON-safe distribution statistics."""
    values = values.detach().double().reshape(-1).cpu()
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")

    quantile_levels = torch.tensor(
        [0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64
    )
    quantiles = torch.quantile(values, quantile_levels)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def per_slot_describe(values):
    """Summarize each K column in a [N, K] tensor."""
    if values.ndim != 2:
        raise ValueError("values must have shape [N, K]")
    return [describe(values[:, slot]) for slot in range(values.shape[1])]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether q_k contains slot-specific signal"
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip_download_root", default=None)
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dataset_name", default="AERI-PEDES")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[2.0, 1.0, 0.5, 0.2],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_args(args):
    return SimpleNamespace(
        loss_names="cda",
        pretrain_choice=args.pretrain_choice,
        clip_download_root=args.clip_download_root,
        img_size=(384, 128),
        stride_size=16,
        temperature=0.02,
        avm_mode="slot",
        avm_num_slots=args.num_slots,
        avm_supervision="qk",
        avm_qk_temperature=1.0,
        avm_mask_loss_weight=0.0,
        avm_loss_weight=1.0,
        img_aug=True,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_length=77,
        pretrain="",
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    refined = {
        key.replace("module.", "", 1): value.detach().clone()
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(refined, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def attention_diagnostics(tokens, slot_pool):
    attention = slot_attention(tokens, slot_pool)
    attention_similarity = pairwise_slot_cosine(attention)
    return (
        off_diagonal_values(attention_similarity),
        normalized_attention_entropy(attention),
    )


def collect_diagnostics(model, train_loader, args):
    matched_batches = []
    mismatched_batches = []
    sample_gap_batches = []
    ground_feature_offdiag = []
    aerial_feature_offdiag = []
    ground_attention_offdiag = []
    aerial_attention_offdiag = []
    ground_attention_entropy = []
    aerial_attention_entropy = []
    mask_batches = []
    processed_batches = 0
    skipped_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= args.batches:
                break

            pids = batch["pids"].to(args.device)
            try:
                wrong_indices = make_pid_mismatch_indices(pids)
            except ValueError:
                skipped_batches += 1
                continue

            aerial_images = batch["images"].to(args.device)
            ground_images = batch["ground_imgs"].to(args.device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.device.startswith("cuda"),
            ):
                aerial_features = model.base_model.encode_image(
                    aerial_images
                )
                ground_features = model.base_model.encode_image(
                    ground_images
                )

            aerial_tokens = aerial_features[:, 1:, :]
            ground_tokens = ground_features[:, 1:, :]
            aerial_slots = model.slot_pool(aerial_tokens)
            ground_slots = model.slot_pool(ground_tokens)

            matched = F.cosine_similarity(
                ground_slots, aerial_slots, dim=-1
            )
            mismatched = F.cosine_similarity(
                ground_slots[wrong_indices], aerial_slots, dim=-1
            )

            matched_batches.append(matched.cpu())
            mismatched_batches.append(mismatched.cpu())
            sample_gap_batches.append(
                (matched - mismatched).mean(dim=1).cpu()
            )

            ground_feature_offdiag.append(
                off_diagonal_values(
                    pairwise_slot_cosine(ground_slots)
                ).cpu()
            )
            aerial_feature_offdiag.append(
                off_diagonal_values(
                    pairwise_slot_cosine(aerial_slots)
                ).cpu()
            )

            ground_attn_sim, ground_attn_entropy = (
                attention_diagnostics(ground_tokens, model.slot_pool)
            )
            aerial_attn_sim, aerial_attn_entropy = (
                attention_diagnostics(aerial_tokens, model.slot_pool)
            )
            ground_attention_offdiag.append(ground_attn_sim.cpu())
            aerial_attention_offdiag.append(aerial_attn_sim.cpu())
            ground_attention_entropy.append(
                ground_attn_entropy.cpu()
            )
            aerial_attention_entropy.append(
                aerial_attn_entropy.cpu()
            )

            aerial_cls = aerial_features[:, 0, :].float()
            mask_batches.append(
                model.avm_mask_head(aerial_cls).cpu()
            )
            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "no batch contained at least two different PIDs"
        )

    matched = torch.cat(matched_batches, dim=0)
    mismatched = torch.cat(mismatched_batches, dim=0)
    sample_gaps = torch.cat(sample_gap_batches, dim=0)
    pair_accuracy = (matched > mismatched).float()

    if sample_gaps.numel() > 1:
        gap_sem = sample_gaps.std(unbiased=True) / math.sqrt(
            sample_gaps.numel()
        )
        gap_ci95 = [
            float(sample_gaps.mean() - 1.96 * gap_sem),
            float(sample_gaps.mean() + 1.96 * gap_sem),
        ]
    else:
        gap_ci95 = [None, None]

    q_by_temperature = {}
    for temperature in args.taus:
        if temperature <= 0:
            raise ValueError("all tau values must be positive")
        q_target = torch.sigmoid(matched / temperature)
        q_by_temperature[str(temperature)] = {
            "overall": describe(q_target),
            "per_slot": per_slot_describe(q_target),
        }

    return {
        "processed_batches": processed_batches,
        "skipped_batches": skipped_batches,
        "samples": int(matched.shape[0]),
        "slots": int(matched.shape[1]),
        "matched_cosine": {
            "overall": describe(matched),
            "per_slot": per_slot_describe(matched),
        },
        "mismatched_cosine": {
            "overall": describe(mismatched),
            "per_slot": per_slot_describe(mismatched),
        },
        "matched_minus_mismatched": {
            "per_sample": describe(sample_gaps),
            "mean_95_percent_ci": gap_ci95,
            "pair_accuracy": float(pair_accuracy.mean()),
            "pair_accuracy_per_slot": [
                float(pair_accuracy[:, slot].mean())
                for slot in range(pair_accuracy.shape[1])
            ],
        },
        "ground_slot_feature_offdiag": describe(
            torch.cat(ground_feature_offdiag, dim=0)
        ),
        "aerial_slot_feature_offdiag": describe(
            torch.cat(aerial_feature_offdiag, dim=0)
        ),
        "ground_slot_attention_offdiag": describe(
            torch.cat(ground_attention_offdiag, dim=0)
        ),
        "aerial_slot_attention_offdiag": describe(
            torch.cat(aerial_attention_offdiag, dim=0)
        ),
        "ground_attention_entropy": describe(
            torch.cat(ground_attention_entropy, dim=0)
        ),
        "aerial_attention_entropy": describe(
            torch.cat(aerial_attention_entropy, dim=0)
        ),
        "initial_mask": describe(torch.cat(mask_batches, dim=0)),
        "q_by_temperature": q_by_temperature,
    }


def print_summary(report):
    matched = report["matched_cosine"]["overall"]
    mismatched = report["mismatched_cosine"]["overall"]
    comparison = report["matched_minus_mismatched"]

    print("\n===== Q_K DIAGNOSTIC SUMMARY =====")
    print("samples:", report["samples"])
    print("slots:", report["slots"])
    print(
        "matched cosine:   "
        f"mean={matched['mean']:.6f} std={matched['std']:.6f}"
    )
    print(
        "mismatched cosine:"
        f" mean={mismatched['mean']:.6f} "
        f"std={mismatched['std']:.6f}"
    )
    print(
        "matched-wrong gap:"
        f" mean={comparison['per_sample']['mean']:.6f} "
        f"accuracy={comparison['pair_accuracy']:.4f}"
    )
    print(
        "ground slot feature offdiag mean:",
        f"{report['ground_slot_feature_offdiag']['mean']:.6f}",
    )
    print(
        "aerial slot feature offdiag mean:",
        f"{report['aerial_slot_feature_offdiag']['mean']:.6f}",
    )
    print(
        "ground slot attention offdiag mean:",
        f"{report['ground_slot_attention_offdiag']['mean']:.6f}",
    )
    print(
        "aerial slot attention offdiag mean:",
        f"{report['aerial_slot_attention_offdiag']['mean']:.6f}",
    )
    for temperature, statistics in report[
        "q_by_temperature"
    ].items():
        overall = statistics["overall"]
        print(
            f"tau={temperature}: q_mean={overall['mean']:.6f} "
            f"q_std={overall['std']:.6f}"
        )


def main():
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("batches must be positive")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_seed(args.seed)

    from datasets.build import build_zero_shot_loader
    from model.build_finetune import build_finetune_model

    model_args = build_model_args(args)
    _, train_loader, _, _, num_classes = build_zero_shot_loader(
        model_args, finetune=True
    )
    model = build_finetune_model(model_args, num_classes)
    checkpoint_report = load_checkpoint(model, args.checkpoint)
    model = model.float().to(args.device)

    report = collect_diagnostics(model, train_loader, args)
    report["configuration"] = {
        "dataset_name": args.dataset_name,
        "root_dir": os.path.abspath(args.root_dir),
        "checkpoint": os.path.abspath(args.checkpoint),
        "pretrain_choice": args.pretrain_choice,
        "clip_download_root": (
            os.path.abspath(args.clip_download_root)
            if args.clip_download_root
            else None
        ),
        "batch_size": args.batch_size,
        "requested_batches": args.batches,
        "num_slots": args.num_slots,
        "taus": args.taus,
        "seed": args.seed,
        "device": args.device,
        "training_or_updates": False,
    }
    report["checkpoint_load"] = checkpoint_report

    output_path = os.path.abspath(args.output_json)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print_summary(report)
    print("report:", output_path)
    print("MODEL_UPDATE: NONE")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup, pipeline

from src.biobert_bc5cdr import chunk_text, predictions_to_entities
from src.experiment_config import PROJECT_ROOT, RAW_DIR as BC5_RAW_DIR, RAW_FILENAMES as BC5_RAW_FILENAMES
from src.parse_bc5cdr import parse_bc5cdr
from src.parse_biored import parse_biored, resolve_input as resolve_biored_input
from src.resource_monitor import ResourceMonitor, append_stage_resource, folder_size_mb
from src.utils import (
    deduplicate_entities,
    exact_per_label_report,
    exact_span_metrics,
    group_by_row,
    load_jsonl,
    metrics_from_counts,
    per_document_exact_counts,
    save_jsonl,
)

MODEL_SPECS = OrderedDict(
    [
        ("biobert", ("Fine-tuned BioBERT", "dmis-lab/biobert-v1.1")),
        ("biomedbert", ("Fine-tuned BiomedBERT", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")),
        ("bert_base", ("Fine-tuned BERT-base", "bert-base-uncased")),
        ("distilbert", ("Fine-tuned DistilBERT-base", "distilbert-base-uncased")),
    ]
)
MODEL_ORDER = tuple(MODEL_SPECS)
TARGETS = ("DISEASE", "CHEMICAL")


def load_fast_tokenizer(source, model_key: str):
    """Load an offset-capable tokenizer for the controlled WordPiece encoders."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    except (ValueError, ImportError, ModuleNotFoundError) as exc:
        print(
            f"[TOKENIZER FALLBACK] AutoTokenizer failed for {source}: "
            f"{type(exc).__name__}: {exc}"
        )
        if model_key == "distilbert":
            tokenizer = DistilBertTokenizerFast.from_pretrained(source)
        else:
            tokenizer = BertTokenizerFast.from_pretrained(source)

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Fast tokenizer with offset mappings is required for {source}."
        )
    return tokenizer
RESULTS_ROOT = PROJECT_ROOT / "results" / "finetuned_models"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "finetuned_models"
PREDICTION_ROOT = PROJECT_ROOT / "data" / "processed" / "finetuned_models"


class FeatureDataset(Dataset):
    def __init__(self, features: list[dict[str, torch.Tensor]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def label_maps(target: str) -> tuple[dict[str, int], dict[int, str]]:
    labels = ["O", f"B-{target}", f"I-{target}"]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_entities(entities: list[dict], target: str | None = None) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entity in deduplicate_entities(entities):
        label = str(entity["label"]).upper()
        if label not in TARGETS:
            continue
        if target is not None and label != target:
            continue
        grouped[int(entity["row_id"])].append(entity)
    for values in grouped.values():
        values.sort(key=lambda item: (int(item["start_char"]), int(item["end_char"]), str(item["label"])))
    return grouped


def validate_gold_spans(docs: list[dict], entities: list[dict], dataset: str, split: str) -> None:
    doc_map = {int(doc["row_id"]): str(doc["full_text"]) for doc in docs}
    grouped = group_entities(entities)
    errors: list[str] = []
    for row_id, spans in grouped.items():
        text = doc_map.get(row_id)
        if text is None:
            errors.append(f"{row_id}: entity references a missing document")
            continue
        previous_end = -1
        for entity in spans:
            start = int(entity["start_char"])
            end = int(entity["end_char"])
            if start < 0 or end <= start or end > len(text):
                errors.append(f"{row_id}: invalid span {start}:{end}")
                continue
            if text[start:end] != str(entity["text"]):
                errors.append(f"{row_id}: text mismatch at {start}:{end}")
            if start < previous_end:
                errors.append(f"{row_id}: overlapping target spans near {start}:{end}")
            previous_end = max(previous_end, end)
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"{dataset} {split} gold is not safe for flat BIO training.\n{preview}")


def build_target_features(
    docs: list[dict],
    entities: list[dict],
    tokenizer,
    *,
    target: str,
    max_length: int,
    stride: int,
) -> list[dict[str, torch.Tensor]]:
    label2id, _ = label_maps(target)
    grouped = group_entities(entities, target)
    features: list[dict[str, torch.Tensor]] = []

    for doc in docs:
        row_id = int(doc["row_id"])
        text = str(doc["full_text"])
        gold = grouped.get(row_id, [])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        for window_index in range(len(encoded["input_ids"])):
            offsets = [tuple(pair) for pair in encoded["offset_mapping"][window_index]]
            attention = encoded["attention_mask"][window_index]
            real_offsets = [(start, end) for (start, end), mask in zip(offsets, attention) if mask and end > start]
            if not real_offsets:
                continue
            window_start = min(start for start, _ in real_offsets)
            window_end = max(end for _, end in real_offsets)
            labels = [label2id["O"] if end > start else -100 for start, end in offsets]

            contained = [
                entity
                for entity in gold
                if int(entity["start_char"]) >= window_start and int(entity["end_char"]) <= window_end
            ]
            truncated = [
                entity
                for entity in gold
                if int(entity["start_char"]) < window_end
                and int(entity["end_char"]) > window_start
                and entity not in contained
            ]

            for entity in truncated:
                start = int(entity["start_char"])
                end = int(entity["end_char"])
                for token_index, (token_start, token_end) in enumerate(offsets):
                    if token_end > token_start and token_start < end and token_end > start:
                        labels[token_index] = -100

            for entity in contained:
                start = int(entity["start_char"])
                end = int(entity["end_char"])
                token_indices = [
                    token_index
                    for token_index, (token_start, token_end) in enumerate(offsets)
                    if token_end > token_start and token_start < end and token_end > start
                ]
                for position, token_index in enumerate(token_indices):
                    prefix = "B" if position == 0 else "I"
                    labels[token_index] = label2id[f"{prefix}-{target}"]

            feature = {
                "input_ids": torch.tensor(encoded["input_ids"][window_index], dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            if "token_type_ids" in encoded:
                feature["token_type_ids"] = torch.tensor(encoded["token_type_ids"][window_index], dtype=torch.long)
            features.append(feature)
    return features


def predict_target(model, tokenizer, docs: list[dict], target: str, device: torch.device) -> tuple[list[dict], float]:
    model.eval()
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        ignore_labels=["O"],
        device=0 if device.type == "cuda" else -1,
    )
    all_entities: list[dict] = []
    started = time.time()
    with torch.inference_mode():
        for doc_number, doc in enumerate(docs, start=1):
            row_id = int(doc["row_id"])
            text = str(doc["full_text"])
            for chunk, char_offset in chunk_text(text, tokenizer):
                predictions = ner(chunk)
                all_entities.extend(predictions_to_entities(predictions, row_id, target, text, char_offset))
            if doc_number % 50 == 0 or doc_number == len(docs):
                print(f"    {target}: predicted {doc_number}/{len(docs)} documents")
    seconds = time.time() - started
    entities = [
        entity
        for entity in deduplicate_entities(all_entities)
        if str(entity["label"]).upper() == target
    ]
    return entities, seconds


def target_metrics(gold: list[dict], pred: list[dict], target: str) -> dict:
    metrics = exact_span_metrics(gold, pred, label=target)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
    }


def metrics_bundle(gold: list[dict], pred: list[dict]) -> dict:
    overall = exact_span_metrics(gold, pred)
    return {
        "precision": float(overall["precision"]),
        "recall": float(overall["recall"]),
        "f1": float(overall["f1"]),
        "tp": int(overall["tp"]),
        "fp": int(overall["fp"]),
        "fn": int(overall["fn"]),
        "per_label": exact_per_label_report(gold, pred),
    }


def load_dataset(dataset: str, split: str) -> tuple[list[dict], list[dict]]:
    if dataset == "bc5cdr":
        raw = BC5_RAW_DIR / BC5_RAW_FILENAMES[split]
        if not raw.exists():
            raise FileNotFoundError(f"Missing BC5CDR {split} file: {raw}")
        return parse_bc5cdr(raw)
    if dataset == "biored":
        path = resolve_biored_input(split)
        docs, gold, _ = parse_biored(path)
        return docs, gold
    raise ValueError(dataset)


def train_target(
    *,
    dataset: str,
    model_key: str,
    display_name: str,
    base_model: str,
    target: str,
    train_docs: list[dict],
    train_gold: list[dict],
    dev_docs: list[dict],
    dev_gold: list[dict],
    args,
) -> dict:
    target_key = target.lower()
    result_dir = RESULTS_ROOT / dataset / model_key / target_key
    checkpoint_dir = CHECKPOINT_ROOT / dataset / model_key / target_key / "best"
    complete_file = result_dir / "training_complete.json"
    if complete_file.exists() and checkpoint_dir.exists() and not args.force:
        existing = json.loads(complete_file.read_text(encoding="utf-8"))
        resource_fields = (
            "training_peak_ram_mb",
            "training_peak_gpu_vram_mb",
            "checkpoint_storage_mb",
            "parameter_count",
        )
        if all(field in existing for field in resource_fields):
            print(f"[SKIP] {dataset} {display_name} {target}: completed resource-aware checkpoint exists")
            return existing
        print(f"[RESOURCE REFRESH] {dataset} {display_name} {target}: retraining once to capture resource telemetry")

    if args.force:
        shutil.rmtree(CHECKPOINT_ROOT / dataset / model_key / target_key, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    label2id, id2label = label_maps(target)
    tokenizer = load_fast_tokenizer(base_model, model_key)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

    features = build_target_features(
        train_docs,
        train_gold,
        tokenizer,
        target=target,
        max_length=args.max_length,
        stride=args.stride,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        features,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = device.type == "cuda" and not args.no_fp16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rows: list[dict] = []
    best_f1 = -1.0
    best_epoch = None
    training_started = time.time()
    training_monitor = ResourceMonitor(
        f"Fine-tuned {dataset} {display_name} {target} training",
        root_pid=os.getpid(),
        metadata={
            "dataset": dataset,
            "model": display_name,
            "model_key": model_key,
            "target": target,
            "resource_kind": "fine_tune_training",
        },
    ).start()
    print(f"\n{dataset.upper()} | {display_name} | {target} | {base_model}")
    print(f"  training windows: {len(features)} | device: {device} | fp16: {use_amp}")

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        dev_pred, _ = predict_target(model, tokenizer, dev_docs, target, device)
        dev_metrics = target_metrics(dev_gold, dev_pred, target)
        row = {
            "dataset": dataset,
            "model": display_name,
            "target": target,
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(loader)),
            "dev_precision": dev_metrics["precision"],
            "dev_recall": dev_metrics["recall"],
            "dev_f1": dev_metrics["f1"],
            "seconds": time.time() - epoch_started,
        }
        rows.append(row)
        print(
            f"  epoch {epoch}: loss={row['train_loss']:.4f} "
            f"DEV P={dev_metrics['precision']:.4f} R={dev_metrics['recall']:.4f} F1={dev_metrics['f1']:.4f}"
        )
        if float(dev_metrics["f1"]) > best_f1 + 1e-12:
            best_f1 = float(dev_metrics["f1"])
            best_epoch = epoch
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            save_json(dev_metrics, result_dir / "best_dev_metrics.json")
            save_jsonl(dev_pred, result_dir / "best_dev_predictions.jsonl")

    write_csv(result_dir / "epoch_metrics.csv", rows)
    training_resource = training_monitor.stop()
    append_stage_resource(training_resource)
    payload = {
        "dataset": dataset,
        "model_key": model_key,
        "model": display_name,
        "base_model": base_model,
        "target": target,
        "training_split": f"official {dataset} train only",
        "selection_split": f"official {dataset} dev only",
        "test_loaded_during_training": False,
        "best_epoch": int(best_epoch),
        "best_dev_f1": float(best_f1),
        "training_seconds": time.time() - training_started,
        "training_gpu_hours": float(training_resource.get("gpu_hours") or 0.0),
        "training_peak_ram_mb": training_resource.get("peak_process_tree_ram_mb"),
        "training_peak_gpu_vram_mb": training_resource.get("peak_process_gpu_vram_mb"),
        "training_peak_total_gpu_memory_used_mb": training_resource.get("peak_total_gpu_memory_used_mb"),
        "training_mean_system_gpu_utilization_percent": training_resource.get("mean_system_gpu_utilization_percent"),
        "training_approx_system_gpu_energy_wh": training_resource.get("approx_system_gpu_energy_wh"),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "checkpoint_storage_mb": float(folder_size_mb(checkpoint_dir)),
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "stride": args.stride,
        "best_checkpoint": str(checkpoint_dir),
    }
    save_json(payload, complete_file)
    del model, tokenizer, loader, features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _aggregate(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        row = counts[int(index)]
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
    return metrics_from_counts(tp, fp, fn)


def _paired_p(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    n = len(difference)
    lo = (np.count_nonzero(difference <= 0) + 1) / (n + 1)
    hi = (np.count_nonzero(difference >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    adjusted = [0.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(raw) - rank) * raw[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def bootstrap_cawv_vs_models(
    docs: list[dict],
    gold: list[dict],
    cawv: list[dict],
    models: dict[str, list[dict]],
    *,
    resamples: int,
    seed: int,
) -> dict:
    gold_by_row = group_by_row(gold)
    all_predictions = {"CAWV": cawv, **models}
    counts = {
        key: per_document_exact_counts(docs, gold_by_row, group_by_row(pred))
        for key, pred in all_predictions.items()
    }
    n_docs = len(docs)
    full = np.arange(n_docs)
    observed = {key: _aggregate(value, full) for key, value in counts.items()}
    samples = {key: np.zeros(resamples, dtype=float) for key in all_predictions}
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        indices = rng.integers(0, n_docs, size=n_docs)
        for key, value in counts.items():
            samples[key][iteration] = float(_aggregate(value, indices)["f1"])
        if (iteration + 1) % 200 == 0:
            print(f"  bootstrap {iteration + 1}/{resamples}")

    intervals = {}
    for key in all_predictions:
        low, high = np.percentile(samples[key], [2.5, 97.5])
        intervals[key] = {
            "observed_f1": float(observed[key]["f1"]),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    rows = []
    raw = []
    for model_key in MODEL_ORDER:
        difference = samples["CAWV"] - samples[model_key]
        ci_low, ci_high = np.percentile(difference, [2.5, 97.5])
        p = _paired_p(samples["CAWV"], samples[model_key])
        raw.append(p)
        rows.append(
            {
                "comparison": f"CAWV_vs_{model_key}",
                "model_key": model_key,
                "f1_cawv": float(observed["CAWV"]["f1"]),
                "f1_model": float(observed[model_key]["f1"]),
                "delta_f1": float(observed["CAWV"]["f1"] - observed[model_key]["f1"]),
                "delta_ci_low": float(ci_low),
                "delta_ci_high": float(ci_high),
                "raw_p": float(p),
            }
        )
    for row, adjusted in zip(rows, _holm(raw)):
        row["holm_p"] = float(adjusted)
        row["significant_holm_0.05"] = bool(adjusted < 0.05)
    return {
        "method": "paired_document_level_bootstrap",
        "resamples": resamples,
        "seed": seed,
        "multiple_testing_correction": "Holm across four CAWV-vs-controlled-model comparisons within the corpus",
        "f1_confidence_intervals_95": intervals,
        "comparisons": rows,
    }


def bootstrap_public_vs_finetuned(
    docs: list[dict],
    gold: list[dict],
    public: list[dict],
    finetuned: list[dict],
    *,
    resamples: int,
    seed: int,
) -> dict:
    gold_by_row = group_by_row(gold)
    p_counts = per_document_exact_counts(docs, gold_by_row, group_by_row(public))
    f_counts = per_document_exact_counts(docs, gold_by_row, group_by_row(finetuned))
    n_docs = len(docs)
    full = np.arange(n_docs)
    observed_public = _aggregate(p_counts, full)
    observed_finetuned = _aggregate(f_counts, full)
    p_samples = np.zeros(resamples)
    f_samples = np.zeros(resamples)
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        indices = rng.integers(0, n_docs, size=n_docs)
        p_samples[iteration] = float(_aggregate(p_counts, indices)["f1"])
        f_samples[iteration] = float(_aggregate(f_counts, indices)["f1"])
    difference = f_samples - p_samples
    ci_low, ci_high = np.percentile(difference, [2.5, 97.5])
    raw_p = _paired_p(f_samples, p_samples)
    return {
        "comparison": "Fine-tuned BioBERT vs public BioBERT",
        "public_f1": float(observed_public["f1"]),
        "finetuned_f1": float(observed_finetuned["f1"]),
        "delta_f1": float(observed_finetuned["f1"] - observed_public["f1"]),
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "raw_p": float(raw_p),
        "resamples": resamples,
        "seed": seed,
    }


def cawv_test_file(dataset: str) -> Path:
    if dataset == "bc5cdr":
        return PROJECT_ROOT / "data" / "gold" / "cawv_pseudo_gold_test_entities_bc5cdr.jsonl"
    return PROJECT_ROOT / "data" / "gold" / "biored" / "cawv_pseudo_gold_test_entities_biored.jsonl"


def public_biobert_file(dataset: str) -> Path:
    if dataset == "bc5cdr":
        return PROJECT_ROOT / "data" / "processed" / "bc5cdr" / "biobert_test_entities_bc5cdr.jsonl"
    return PROJECT_ROOT / "data" / "processed" / "biored" / "biobert_test_entities_biored.jsonl"


def plot_performance(dataset: str, cawv_metrics: dict, rows: list[dict], path: Path) -> None:
    labels = ["CAWV"] + [row["model"].replace("Fine-tuned ", "") for row in rows]
    data = [
        {"precision": cawv_metrics["precision"], "recall": cawv_metrics["recall"], "f1": cawv_metrics["f1"]},
        *rows,
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for index, metric in enumerate(("precision", "recall", "f1")):
        values = [float(row[metric]) for row in data]
        ax.bar(x + (index - 1) * width, values, width, label=metric.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact-span score")
    title_name = "BC5CDR" if dataset == "bc5cdr" else "BioRED"
    ax.set_title(f"CAWV and controlled fine-tuned references on {title_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_dataset(dataset: str, args) -> tuple[list[dict], list[dict]]:
    dataset_dir = RESULTS_ROOT / dataset
    prediction_dir = PREDICTION_ROOT / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}\nCONTROLLED FINE-TUNING: {dataset.upper()}\n{'=' * 78}")
    print("Loading TRAIN and DEV only. TEST remains unopened during training/model selection.")
    train_docs, train_gold = load_dataset(dataset, "train")
    dev_docs, dev_gold = load_dataset(dataset, "dev")
    validate_gold_spans(train_docs, train_gold, dataset, "train")
    validate_gold_spans(dev_docs, dev_gold, dataset, "dev")

    manifests = []
    for model_key, (display_name, base_model) in MODEL_SPECS.items():
        target_runs = []
        for target in TARGETS:
            target_runs.append(
                train_target(
                    dataset=dataset,
                    model_key=model_key,
                    display_name=display_name,
                    base_model=base_model,
                    target=target,
                    train_docs=train_docs,
                    train_gold=train_gold,
                    dev_docs=dev_docs,
                    dev_gold=dev_gold,
                    args=args,
                )
            )
        manifests.append({"model_key": model_key, "model": display_name, "base_model": base_model, "targets": target_runs})

    save_json(
        {
            "dataset": dataset,
            "protocol": "two independently fine-tuned token-classification heads per encoder; TRAIN updates parameters; DEV independently selects disease and chemical epochs; TEST loaded only after selection",
            "test_used_for_training_or_selection": False,
            "models": manifests,
        },
        dataset_dir / "training_manifest.json",
    )

    print("All disease and chemical checkpoints are frozen. Loading TEST now.")
    test_docs, test_gold = load_dataset(dataset, "test")
    validate_gold_spans(test_docs, test_gold, dataset, "test")

    performance_rows = []
    merged_predictions: dict[str, list[dict]] = {}
    for model_key, (display_name, _base_model) in MODEL_SPECS.items():
        model_result_dir = dataset_dir / model_key
        merged_file = prediction_dir / f"{model_key}_test_entities.jsonl"
        metric_file = model_result_dir / "test_metrics.json"
        if merged_file.exists() and metric_file.exists() and not args.force:
            merged = load_jsonl(merged_file)
            payload = json.loads(metric_file.read_text(encoding="utf-8"))
            required_resource_fields = (
                "end_to_end_inference_seconds",
                "peak_inference_ram_mb",
                "peak_inference_gpu_vram_mb",
                "checkpoint_storage_mb",
            )
            if all(field in payload for field in required_resource_fields):
                print(f"[SKIP] {dataset} {display_name} TEST inference")
            else:
                print(f"[RESOURCE REFRESH] {dataset} {display_name} TEST inference")
                merged = None
        else:
            merged = None

        if merged is None:
            target_predictions = []
            total_inference = 0.0
            total_end_to_end_inference = 0.0
            inference_resources = []
            checkpoint_sizes = {}
            parameter_counts = {}
            for target in TARGETS:
                checkpoint = CHECKPOINT_ROOT / dataset / model_key / target.lower() / "best"
                monitor = ResourceMonitor(
                    f"Fine-tuned {dataset} {display_name} {target} test inference",
                    root_pid=os.getpid(),
                    metadata={
                        "dataset": dataset,
                        "model": display_name,
                        "model_key": model_key,
                        "target": target,
                        "resource_kind": "fine_tune_test_inference",
                    },
                ).start()
                tokenizer = load_fast_tokenizer(checkpoint, model_key)
                model = AutoModelForTokenClassification.from_pretrained(checkpoint)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                parameter_counts[target] = int(sum(parameter.numel() for parameter in model.parameters()))
                checkpoint_sizes[target] = float(folder_size_mb(checkpoint))
                pred, seconds = predict_target(model, tokenizer, test_docs, target, device)
                save_jsonl(pred, prediction_dir / f"{model_key}_{target.lower()}_test_entities.jsonl")
                target_predictions.extend(pred)
                total_inference += seconds
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                resource = monitor.stop()
                append_stage_resource(resource)
                inference_resources.append(resource)
                total_end_to_end_inference += float(resource.get("wall_seconds") or 0.0)
            merged = deduplicate_entities(target_predictions)
            save_jsonl(merged, merged_file)
            payload = metrics_bundle(test_gold, merged)
            payload.update(
                {
                    "model": display_name,
                    "model_key": model_key,
                    "inference_seconds": float(total_inference),
                    "average_seconds_per_document": float(total_inference / max(1, len(test_docs))),
                    "end_to_end_inference_seconds": float(total_end_to_end_inference),
                    "end_to_end_seconds_per_document": float(total_end_to_end_inference / max(1, len(test_docs))),
                    "peak_inference_ram_mb": max(
                        [float(row["peak_process_tree_ram_mb"]) for row in inference_resources if row.get("peak_process_tree_ram_mb") is not None],
                        default=None,
                    ),
                    "peak_inference_gpu_vram_mb": max(
                        [float(row["peak_process_gpu_vram_mb"]) for row in inference_resources if row.get("peak_process_gpu_vram_mb") is not None],
                        default=None,
                    ),
                    "peak_total_gpu_memory_used_mb": max(
                        [float(row["peak_total_gpu_memory_used_mb"]) for row in inference_resources if row.get("peak_total_gpu_memory_used_mb") is not None],
                        default=None,
                    ),
                    "approx_system_gpu_energy_wh": sum(
                        float(row.get("approx_system_gpu_energy_wh") or 0.0) for row in inference_resources
                    ) or None,
                    "gpu_hours_test_inference": float(total_end_to_end_inference / 3600.0),
                    "checkpoint_count": 2,
                    "checkpoint_storage_mb": float(sum(checkpoint_sizes.values())),
                    "disease_checkpoint_storage_mb": float(checkpoint_sizes.get("DISEASE", 0.0)),
                    "chemical_checkpoint_storage_mb": float(checkpoint_sizes.get("CHEMICAL", 0.0)),
                    "parameter_count_per_head": int(max(parameter_counts.values())) if parameter_counts else None,
                    "two_checkpoint_setup": True,
                }
            )
            save_json(payload, metric_file)
        merged_predictions[model_key] = merged
        performance_rows.append(
            {
                "dataset": dataset,
                "model": display_name,
                "model_key": model_key,
                "precision": float(payload["precision"]),
                "recall": float(payload["recall"]),
                "f1": float(payload["f1"]),
                "tp": int(payload["tp"]),
                "fp": int(payload["fp"]),
                "fn": int(payload["fn"]),
                "inference_seconds": float(payload.get("inference_seconds", 0.0)),
                "average_seconds_per_document": float(payload.get("average_seconds_per_document", 0.0)),
                "end_to_end_inference_seconds": float(payload.get("end_to_end_inference_seconds", payload.get("inference_seconds", 0.0))),
                "end_to_end_seconds_per_document": float(payload.get("end_to_end_seconds_per_document", payload.get("average_seconds_per_document", 0.0))),
                "peak_inference_ram_mb": payload.get("peak_inference_ram_mb"),
                "peak_inference_gpu_vram_mb": payload.get("peak_inference_gpu_vram_mb"),
                "peak_total_gpu_memory_used_mb": payload.get("peak_total_gpu_memory_used_mb"),
                "approx_system_gpu_energy_wh": payload.get("approx_system_gpu_energy_wh"),
                "gpu_hours_test_inference": payload.get("gpu_hours_test_inference"),
                "checkpoint_count": int(payload.get("checkpoint_count", 2)),
                "checkpoint_storage_mb": payload.get("checkpoint_storage_mb"),
                "parameter_count_per_head": payload.get("parameter_count_per_head"),
                "two_checkpoint_setup": bool(payload.get("two_checkpoint_setup", True)),
            }
        )

    cawv_path = cawv_test_file(dataset)
    if not cawv_path.exists():
        raise FileNotFoundError(f"Missing CAWV TEST predictions for supervised comparison: {cawv_path}")
    cawv = load_jsonl(cawv_path)
    cawv_metrics = metrics_bundle(test_gold, cawv)
    bootstrap = bootstrap_cawv_vs_models(
        test_docs,
        test_gold,
        cawv,
        merged_predictions,
        resamples=args.resamples,
        seed=args.seed,
    )
    for row in performance_rows:
        ci = bootstrap["f1_confidence_intervals_95"][row["model_key"]]
        row["f1_ci_low"] = ci["ci_low"]
        row["f1_ci_high"] = ci["ci_high"]
        comparison = next(item for item in bootstrap["comparisons"] if item["model_key"] == row["model_key"])
        row["cawv_minus_model_f1"] = comparison["delta_f1"]
        row["comparison_ci_low"] = comparison["delta_ci_low"]
        row["comparison_ci_high"] = comparison["delta_ci_high"]
        row["raw_p"] = comparison["raw_p"]
        row["holm_p"] = comparison["holm_p"]

    cawv_ci = bootstrap["f1_confidence_intervals_95"]["CAWV"]
    save_json(
        {
            "dataset": dataset,
            "cawv": {**cawv_metrics, "ci_low": cawv_ci["ci_low"], "ci_high": cawv_ci["ci_high"]},
            "models": performance_rows,
        },
        dataset_dir / "performance.json",
    )
    write_csv(dataset_dir / "performance.csv", performance_rows)
    save_json(bootstrap, dataset_dir / "cawv_vs_finetuned_bootstrap.json")

    comparison_rows = []
    for row in bootstrap["comparisons"]:
        comparison_rows.append(
            {
                "dataset": dataset,
                "comparison": f"CAWV vs {MODEL_SPECS[row['model_key']][0].replace('Fine-tuned ', '')}",
                "model_key": row["model_key"],
                "delta_f1": row["delta_f1"],
                "delta_ci_low": row["delta_ci_low"],
                "delta_ci_high": row["delta_ci_high"],
                "raw_p": row["raw_p"],
                "holm_p": row["holm_p"],
                "significant_holm_0.05": row["significant_holm_0.05"],
            }
        )
    write_csv(dataset_dir / "cawv_vs_finetuned_bootstrap.csv", comparison_rows)
    plot_performance(dataset, cawv_metrics, performance_rows, dataset_dir / "finetuned_model_comparison.png")

    public_path = public_biobert_file(dataset)
    if public_path.exists():
        public_cmp = bootstrap_public_vs_finetuned(
            test_docs,
            test_gold,
            load_jsonl(public_path),
            merged_predictions["biobert"],
            resamples=args.resamples,
            seed=args.seed,
        )
        save_json(public_cmp, dataset_dir / "public_vs_finetuned_biobert.json")
        write_csv(dataset_dir / "public_vs_finetuned_biobert.csv", [{"dataset": dataset, **public_cmp}])
    else:
        print(f"[NOTE] Public BioBERT predictions not found for {dataset}; public-vs-fine-tuned comparison skipped.")

    return performance_rows, comparison_rows


def run_all(*, force: bool = False, resamples: int = 1000, seed: int = 42, **overrides) -> Path:
    class Args:
        pass

    args = Args()
    args.force = force
    args.resamples = resamples
    args.seed = seed
    args.epochs = int(overrides.get("epochs", 5))
    args.learning_rate = float(overrides.get("learning_rate", 5e-5))
    args.weight_decay = float(overrides.get("weight_decay", 0.01))
    args.warmup_ratio = float(overrides.get("warmup_ratio", 0.1))
    args.train_batch_size = int(overrides.get("train_batch_size", 2))
    args.gradient_accumulation_steps = int(overrides.get("gradient_accumulation_steps", 4))
    args.max_length = int(overrides.get("max_length", 512))
    args.stride = int(overrides.get("stride", 128))
    args.max_grad_norm = float(overrides.get("max_grad_norm", 1.0))
    args.no_fp16 = bool(overrides.get("no_fp16", False))

    all_performance = []
    all_comparisons = []
    for dataset in ("bc5cdr", "biored"):
        performance, comparisons = evaluate_dataset(dataset, args)
        all_performance.extend(performance)
        all_comparisons.extend(comparisons)

    write_csv(RESULTS_ROOT / "combined_performance.csv", all_performance)
    write_csv(RESULTS_ROOT / "combined_cawv_vs_finetuned_bootstrap.csv", all_comparisons)

    perf_lookup = {
        dataset: {row["model_key"]: row for row in all_performance if row["dataset"] == dataset}
        for dataset in ("bc5cdr", "biored")
    }
    symmetric_performance = []
    for model_key in MODEL_ORDER:
        bc = perf_lookup["bc5cdr"][model_key]
        br = perf_lookup["biored"][model_key]
        symmetric_performance.append(
            {
                "model": MODEL_SPECS[model_key][0],
                "bc5cdr_precision": bc["precision"],
                "bc5cdr_recall": bc["recall"],
                "bc5cdr_f1": bc["f1"],
                "bc5cdr_f1_ci_low": bc["f1_ci_low"],
                "bc5cdr_f1_ci_high": bc["f1_ci_high"],
                "bc5cdr_cawv_minus_model_f1": bc["cawv_minus_model_f1"],
                "bc5cdr_delta_ci_low": bc["comparison_ci_low"],
                "bc5cdr_delta_ci_high": bc["comparison_ci_high"],
                "bc5cdr_raw_p": bc["raw_p"],
                "bc5cdr_holm_p": bc["holm_p"],
                "bc5cdr_inference_seconds": bc["inference_seconds"],
                "bc5cdr_end_to_end_inference_seconds": bc["end_to_end_inference_seconds"],
                "bc5cdr_peak_ram_mb": bc["peak_inference_ram_mb"],
                "bc5cdr_peak_gpu_vram_mb": bc["peak_inference_gpu_vram_mb"],
                "bc5cdr_checkpoint_storage_mb": bc["checkpoint_storage_mb"],
                "bc5cdr_checkpoint_count": bc["checkpoint_count"],
                "biored_precision": br["precision"],
                "biored_recall": br["recall"],
                "biored_f1": br["f1"],
                "biored_f1_ci_low": br["f1_ci_low"],
                "biored_f1_ci_high": br["f1_ci_high"],
                "biored_cawv_minus_model_f1": br["cawv_minus_model_f1"],
                "biored_delta_ci_low": br["comparison_ci_low"],
                "biored_delta_ci_high": br["comparison_ci_high"],
                "biored_raw_p": br["raw_p"],
                "biored_holm_p": br["holm_p"],
                "biored_inference_seconds": br["inference_seconds"],
                "biored_end_to_end_inference_seconds": br["end_to_end_inference_seconds"],
                "biored_peak_ram_mb": br["peak_inference_ram_mb"],
                "biored_peak_gpu_vram_mb": br["peak_inference_gpu_vram_mb"],
                "biored_checkpoint_storage_mb": br["checkpoint_storage_mb"],
                "biored_checkpoint_count": br["checkpoint_count"],
            }
        )
    write_csv(RESULTS_ROOT / "symmetric_performance_and_statistics.csv", symmetric_performance)

    public_rows = []
    for dataset in ("bc5cdr", "biored"):
        path = RESULTS_ROOT / dataset / "public_vs_finetuned_biobert.json"
        if path.exists():
            public_rows.append({"dataset": dataset, **json.loads(path.read_text(encoding="utf-8"))})
    if len(public_rows) == 2:
        bc = next(row for row in public_rows if row["dataset"] == "bc5cdr")
        br = next(row for row in public_rows if row["dataset"] == "biored")
        write_csv(
            RESULTS_ROOT / "symmetric_public_vs_finetuned_biobert.csv",
            [
                {
                    "comparison": "Fine-tuned BioBERT vs public BioBERT",
                    "bc5cdr_public_f1": bc["public_f1"],
                    "bc5cdr_finetuned_f1": bc["finetuned_f1"],
                    "bc5cdr_delta_f1": bc["delta_f1"],
                    "bc5cdr_ci_low": bc["delta_ci_low"],
                    "bc5cdr_ci_high": bc["delta_ci_high"],
                    "bc5cdr_p": bc["raw_p"],
                    "biored_public_f1": br["public_f1"],
                    "biored_finetuned_f1": br["finetuned_f1"],
                    "biored_delta_f1": br["delta_f1"],
                    "biored_ci_low": br["delta_ci_low"],
                    "biored_ci_high": br["delta_ci_high"],
                    "biored_p": br["raw_p"],
                }
            ],
        )

    from src.manuscript_results_log import generate as generate_manuscript_log

    generate_manuscript_log()
    print(f"\nFine-tuned comparison complete: {RESULTS_ROOT}")
    return RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BioBERT, BiomedBERT, BERT-base, and DistilBERT independently on BC5CDR and BioRED."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all(
        force=args.force,
        resamples=args.resamples,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        stride=args.stride,
        max_grad_norm=args.max_grad_norm,
        no_fp16=args.no_fp16,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a lightweight physics-guided instance height calibrator.

The calibrator is designed for weak building-height labels. It keeps the
stereo-derived differential-disparity height as the physical prior and learns a
small residual correction from training instances only. Validation/test labels
are never used to fit the physical scale, feature normalization, or residual
adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


USE_COLUMNS = [
    "inst_id",
    "city",
    "core_name",
    "gt_mean",
    "pixel_count",
    "height_physical_mean",
    "height_physical_p50",
    "height_reverse_mean",
    "delta_abs_mean",
    "delta_abs_p50",
    "delta_relu_mean",
    "delta_signed_mean",
    "disp_mean",
    "disp_p50",
    "ground_disp",
    "ground_disp_image_median",
    "ground_disp_image_mad",
    "disp_order_gap",
    "ground_corrected",
    "disp_order_bad",
    "k",
]

HEIGHT_BINS = [12.0, 24.0, 30.0, 50.0, 100.0, 150.0, 300.0]


@dataclass
class Normalizer:
    mean: List[float]
    std: List[float]
    feature_names: List[str]
    city_to_idx: Dict[str, int]


class InstanceFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, physical: np.ndarray, y: np.ndarray, city_idx: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.physical = torch.from_numpy(physical.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.city_idx = torch.from_numpy(city_idx.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "physical": self.physical[idx],
            "y": self.y[idx],
            "city_idx": self.city_idx[idx],
        }


class PhysicsGuidedInstanceCalibrator(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_cities: int,
        use_city_embedding: bool,
        init_scale: float,
        init_bias: float,
        hidden_dim: int,
        layers: int,
        dropout: float,
        residual_clip: float,
        low_residual_clip: float,
    ):
        super().__init__()
        self.use_city_embedding = bool(use_city_embedding)
        self.residual_clip = float(residual_clip)
        self.low_residual_clip = float(low_residual_clip)
        self.physical_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.physical_bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

        emb_dim = 6 if self.use_city_embedding else 0
        self.city_embedding = nn.Embedding(max(num_cities, 1), emb_dim) if self.use_city_embedding else None
        in_dim = int(num_features) + emb_dim
        blocks: List[nn.Module] = []
        dim = in_dim
        for _ in range(max(int(layers), 1)):
            blocks.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.residual_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, features: torch.Tensor, physical: torch.Tensor, city_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features
        if self.city_embedding is not None:
            emb = self.city_embedding(torch.clamp(city_idx, 0, self.city_embedding.num_embeddings - 1))
            x = torch.cat([x, emb], dim=-1)
        base = self.physical_scale * physical + self.physical_bias
        raw = self.residual_head(self.trunk(x)).squeeze(-1)

        high_gate = torch.sigmoid((physical - 12.0) / 6.0)
        clip = self.low_residual_clip * (1.0 - high_gate) + self.residual_clip * high_gate
        residual = clip * torch.tanh(raw)
        pred = torch.clamp(base + residual, 3.0, 260.0)
        return pred, base, residual


def _load_manifest(path: Path) -> Tuple[set[str], Dict[str, set[int]]]:
    payload = json.loads(path.read_text())
    valid_tiles = {str(x) for x in payload.get("valid_tiles", [])}
    ignore = {
        str(tile): {int(v) for v in ids}
        for tile, ids in payload.get("ignore_instances", {}).items()
    }
    return valid_tiles, ignore


def _apply_manifest(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    valid_tiles, ignore = _load_manifest(manifest_path)
    tile_key = df["city"].astype(str) + "/" + df["core_name"].astype(str)
    sub = df.loc[tile_key.isin(valid_tiles)].copy()
    if sub.empty:
        return sub
    sub_tile = (sub["city"].astype(str) + "/" + sub["core_name"].astype(str)).to_numpy()
    inst = sub["inst_id"].astype(int).to_numpy()
    keep = np.asarray([int(i) not in ignore.get(str(t), set()) for t, i in zip(sub_tile, inst)], dtype=bool)
    return sub.loc[keep].copy().reset_index(drop=True)


def _as_bool_float(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in USE_COLUMNS:
        if col not in ("city", "core_name", "ground_corrected", "disp_order_bad"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["log_area"] = np.log1p(out["pixel_count"].clip(lower=0.0))
    out["sqrt_area"] = np.sqrt(out["pixel_count"].clip(lower=0.0))
    out["phys_area"] = out["height_physical_mean"] * out["log_area"]
    out["phys_sqrt_area"] = out["height_physical_mean"] * out["sqrt_area"]
    out["phys2"] = out["height_physical_mean"] * out["height_physical_mean"]
    out["delta2"] = out["delta_abs_mean"] * out["delta_abs_mean"]
    out["sqrt_phys"] = np.sqrt(out["height_physical_mean"].clip(lower=0.0) + 1.0e-6)
    out["abs_signed_delta"] = out["delta_signed_mean"].abs()
    out["ground_minus_median"] = out["ground_disp"] - out["ground_disp_image_median"]
    out["phys_delta_ratio"] = out["height_physical_mean"] / (out["delta_abs_mean"].abs() + 1.0)
    out["ground_corrected_f"] = _as_bool_float(out["ground_corrected"])
    out["disp_order_bad_f"] = _as_bool_float(out["disp_order_bad"])
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_names() -> List[str]:
    return [
        "height_physical_mean",
        "height_physical_p50",
        "height_reverse_mean",
        "delta_abs_mean",
        "delta_abs_p50",
        "delta_relu_mean",
        "delta_signed_mean",
        "abs_signed_delta",
        "disp_mean",
        "disp_p50",
        "ground_disp",
        "ground_disp_image_median",
        "ground_disp_image_mad",
        "ground_minus_median",
        "disp_order_gap",
        "k",
        "log_area",
        "sqrt_area",
        "phys_area",
        "phys_sqrt_area",
        "phys2",
        "delta2",
        "sqrt_phys",
        "phys_delta_ratio",
        "ground_corrected_f",
        "disp_order_bad_f",
    ]


def _build_arrays(train_df: pd.DataFrame, val_df: pd.DataFrame, use_city_embedding: bool) -> Tuple[InstanceFeatureDataset, InstanceFeatureDataset, Normalizer]:
    train = _add_features(train_df)
    val = _add_features(val_df)
    names = _feature_names()
    train_x = train[names].to_numpy(dtype=np.float64)
    val_x = val[names].to_numpy(dtype=np.float64)
    med = np.nanmedian(train_x, axis=0)
    train_x = np.where(np.isfinite(train_x), train_x, med)
    val_x = np.where(np.isfinite(val_x), val_x, med)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    if use_city_embedding:
        city_to_idx = {city: i for i, city in enumerate(sorted(train["city"].astype(str).unique().tolist()))}
        unknown_idx = len(city_to_idx)
        city_to_idx["<UNK>"] = unknown_idx
        train_city = train["city"].astype(str).map(city_to_idx).fillna(unknown_idx).to_numpy(dtype=np.int64)
        val_city = val["city"].astype(str).map(city_to_idx).fillna(unknown_idx).to_numpy(dtype=np.int64)
    else:
        city_to_idx = {"<NONE>": 0}
        train_city = np.zeros(len(train), dtype=np.int64)
        val_city = np.zeros(len(val), dtype=np.int64)

    train_ds = InstanceFeatureDataset(
        train_x,
        train["height_physical_mean"].to_numpy(dtype=np.float64),
        train["gt_mean"].to_numpy(dtype=np.float64),
        train_city,
    )
    val_ds = InstanceFeatureDataset(
        val_x,
        val["height_physical_mean"].to_numpy(dtype=np.float64),
        val["gt_mean"].to_numpy(dtype=np.float64),
        val_city,
    )
    normalizer = Normalizer(mean=mean.tolist(), std=std.tolist(), feature_names=names, city_to_idx=city_to_idx)
    return train_ds, val_ds, normalizer


def _height_bins(y: np.ndarray) -> np.ndarray:
    return np.digitize(y.astype(np.float64), np.asarray(HEIGHT_BINS, dtype=np.float64), right=True)


def _sample_weights(y: np.ndarray) -> np.ndarray:
    bins = _height_bins(y)
    weights = np.ones_like(y, dtype=np.float64)
    for b in np.unique(bins):
        mask = bins == b
        weights[mask] = 1.0 / math.sqrt(float(mask.sum()))
    weights /= max(float(weights.mean()), 1.0e-6)
    high_boost = 1.0 + 0.025 * np.clip(y - 24.0, 0.0, 80.0)
    return np.clip(weights * high_boost, 0.25, 10.0)


def _fit_affine_init(physical: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    model = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1.0e-4, max_iter=1000))
    model.fit(physical.reshape(-1, 1), y, huberregressor__sample_weight=_sample_weights(y))
    scaler = model.named_steps["standardscaler"]
    huber = model.named_steps["huberregressor"]
    scale = float(huber.coef_[0] / scaler.scale_[0])
    bias = float(huber.intercept_ - huber.coef_[0] * scaler.mean_[0] / scaler.scale_[0])
    return scale, bias


def _loss_fn(
    pred: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    y: torch.Tensor,
    soft_abs_margin: float,
    soft_rel_margin: float,
    soft_margin_max: float,
    high_under_weight: float,
    residual_l1_weight: float,
) -> torch.Tensor:
    err = pred - y
    tol = soft_abs_margin + soft_rel_margin * y.abs()
    if soft_margin_max > 0:
        tol = torch.clamp(tol, max=soft_margin_max)
    soft_err = torch.sign(err) * F.relu(err.abs() - tol)
    beta = 2.5
    main = F.smooth_l1_loss(soft_err, torch.zeros_like(soft_err), beta=beta, reduction="none")

    bins = torch.bucketize(y, torch.tensor(HEIGHT_BINS, device=y.device, dtype=y.dtype), right=True)
    bin_weights = torch.ones_like(y)
    bin_weights = torch.where(bins == 0, torch.full_like(bin_weights, 0.65), bin_weights)
    bin_weights = torch.where(bins == 1, torch.full_like(bin_weights, 0.95), bin_weights)
    bin_weights = torch.where(bins == 2, torch.full_like(bin_weights, 1.45), bin_weights)
    bin_weights = torch.where(bins >= 3, torch.full_like(bin_weights, 3.2), bin_weights)
    under = F.relu(y - pred)
    high_under = torch.where(y >= 30.0, under, torch.zeros_like(under))
    low_over = torch.where(y < 12.0, F.relu(pred - y), torch.zeros_like(pred))
    return (
        (main * bin_weights).mean()
        + high_under_weight * F.smooth_l1_loss(high_under, torch.zeros_like(high_under), beta=4.0)
        + 0.25 * F.smooth_l1_loss(low_over, torch.zeros_like(low_over), beta=2.0)
        + residual_l1_weight * residual.abs().mean()
        + 0.002 * (pred - base).pow(2).mean()
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _evaluate(model: nn.Module, ds: InstanceFeatureDataset, batch_size: int, device: torch.device) -> Tuple[dict, np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            physical = batch["physical"].to(device)
            city_idx = batch["city_idx"].to(device)
            pred, _base, _res = model(features, physical, city_idx)
            preds.append(pred.float().cpu().numpy())
            gts.append(batch["y"].float().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    return _metrics(y_true, y_pred), y_true, y_pred


def _print_rows(title: str, rows: Iterable[Tuple[str, dict]]) -> None:
    print(f"\n[{title}]")
    print(f"{'name':<22} {'N':>9} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'R2':>8}")
    print("-" * 72)
    for name, m in rows:
        print(f"{name:<22} {m['n']:>9d} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['bias']:>+8.3f} {m['r2']:>8.3f}")


def _group_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]]]:
    city_rows = []
    for city, sub in df.reset_index(drop=True).groupby("city", sort=True):
        idx = sub.index.to_numpy()
        city_rows.append((str(city), _metrics(y_true[idx], y_pred[idx])))
    bin_rows = []
    bins = _height_bins(y_true)
    for i in range(len(HEIGHT_BINS) + 1):
        mask = bins == i
        if i == 0:
            name = f"0-{HEIGHT_BINS[0]:g}m"
        elif i < len(HEIGHT_BINS):
            name = f"{HEIGHT_BINS[i - 1]:g}-{HEIGHT_BINS[i]:g}m"
        else:
            name = f">{HEIGHT_BINS[-1]:g}m"
        bin_rows.append((name, _metrics(y_true[mask], y_pred[mask]) if mask.any() else {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan")}))
    return city_rows, bin_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=724)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--val_batch_size", type=int, default=32768)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--residual_clip", type=float, default=22.0)
    parser.add_argument("--low_residual_clip", type=float, default=6.0)
    parser.add_argument("--high_under_weight", type=float, default=0.10)
    parser.add_argument("--residual_l1_weight", type=float, default=0.01)
    parser.add_argument("--soft_abs_margin", type=float, default=1.0)
    parser.add_argument("--soft_rel_margin", type=float, default=0.08)
    parser.add_argument("--soft_margin_max", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--use_city_embedding", action="store_true")
    parser.add_argument("--no_weighted_sampler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    df = pd.read_csv(args.metrics_csv, usecols=USE_COLUMNS)
    train_df = _apply_manifest(df, args.train_manifest)
    val_df = _apply_manifest(df, args.val_manifest)
    if train_df.empty or val_df.empty:
        raise SystemExit(f"Empty split: train={len(train_df)} val={len(val_df)}")

    train_ds, val_ds, normalizer = _build_arrays(train_df, val_df, args.use_city_embedding)
    init_scale, init_bias = _fit_affine_init(
        train_df["height_physical_mean"].to_numpy(dtype=np.float64),
        train_df["gt_mean"].to_numpy(dtype=np.float64),
    )
    model = PhysicsGuidedInstanceCalibrator(
        num_features=train_ds.features.shape[1],
        num_cities=max(normalizer.city_to_idx.values()) + 1,
        use_city_embedding=args.use_city_embedding,
        init_scale=init_scale,
        init_bias=init_bias,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        residual_clip=args.residual_clip,
        low_residual_clip=args.low_residual_clip,
    ).to(device)

    sample_weights = _sample_weights(train_df["gt_mean"].to_numpy(dtype=np.float64))
    if args.no_weighted_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    else:
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights.astype(np.float64)),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "init_scale": float(init_scale),
            "init_bias": float(init_bias),
            "train_instances": int(len(train_df)),
            "val_instances": int(len(val_df)),
            "train_cities": sorted(train_df["city"].astype(str).unique().tolist()),
            "val_cities": sorted(val_df["city"].astype(str).unique().tolist()),
            "normalizer": asdict(normalizer),
        }
    )
    (args.save_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"Train instances: {len(train_df):,} | Val instances: {len(val_df):,}")
    print(f"Train cities: {config['train_cities']}")
    print(f"Val cities:   {config['val_cities']}")
    print(f"Initial physical affine: height = {init_scale:.6f} * physical + {init_bias:.6f}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_score = float("inf")
    best_payload = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            physical = batch["physical"].to(device)
            city_idx = batch["city_idx"].to(device)
            y = batch["y"].to(device)
            pred, base, residual = model(features, physical, city_idx)
            loss = _loss_fn(
                pred,
                base,
                residual,
                y,
                soft_abs_margin=args.soft_abs_margin,
                soft_rel_margin=args.soft_rel_margin,
                soft_margin_max=args.soft_margin_max,
                high_under_weight=args.high_under_weight,
                residual_l1_weight=args.residual_l1_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_metrics, y_val, pred_val = _evaluate(model, val_ds, args.val_batch_size, device)
        train_metrics, _y_train, _pred_train = _evaluate(model, train_ds, args.val_batch_size, device)
        high_mask = y_val >= 30.0
        high_metrics = _metrics(y_val[high_mask], pred_val[high_mask]) if high_mask.any() else {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan")}
        score = val_metrics["rmse"] + 0.15 * max(0.0, -val_metrics["r2"]) + 0.05 * max(0.0, high_metrics["rmse"] - 12.0)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | "
                f"train MAE/RMSE/R2={train_metrics['mae']:.3f}/{train_metrics['rmse']:.3f}/{train_metrics['r2']:.3f} | "
                f"val MAE/RMSE/R2={val_metrics['mae']:.3f}/{val_metrics['rmse']:.3f}/{val_metrics['r2']:.3f} | "
                f"high>=30 RMSE/Bias={high_metrics['rmse']:.3f}/{high_metrics['bias']:+.3f}"
            )

        if score < best_score - 1.0e-4:
            best_score = score
            bad = 0
            city_rows, bin_rows = _group_metrics(val_df, y_val, pred_val)
            best_payload = {
                "epoch": int(epoch),
                "score": float(score),
                "train": train_metrics,
                "val": val_metrics,
                "high_ge_30": high_metrics,
                "city_metrics": {k: v for k, v in city_rows},
                "height_bin_metrics": {k: v for k, v in bin_rows},
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "metrics": best_payload,
                },
                args.save_dir / "best_instance_calibrator.pth",
            )
            (args.save_dir / "best_metrics.json").write_text(json.dumps(best_payload, indent=2))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_payload['epoch'] if best_payload else 'NA'}")
                break

    if best_payload is None:
        raise RuntimeError("No best checkpoint was saved.")

    print("\n[Best validation]")
    _print_rows("overall", [("train", best_payload["train"]), ("val", best_payload["val"]), ("high>=30", best_payload["high_ge_30"])])
    _print_rows("city", best_payload["city_metrics"].items())
    _print_rows("height bins", best_payload["height_bin_metrics"].items())
    print(f"\nSaved best checkpoint: {args.save_dir / 'best_instance_calibrator.pth'}")
    print(f"Saved best metrics:    {args.save_dir / 'best_metrics.json'}")


if __name__ == "__main__":
    main()

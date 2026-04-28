from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim

from src.autoencoder import AutoEncoder
from src.config import parse_config, resolve_run_config
from src.data_manager import DataManager
from src.hash import make_train_hash
from src.utils import set_seed


SEEDS = [40, 41, 42, 43, 44, 45]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)

    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    experiment_config = parse_config(args.config)

    for run_config in experiment_config.runs:
        run_config = resolve_run_config(run_config)
        train_hash = make_train_hash(run_config)

        for seed in args.seeds:
            set_seed(seed)

            autoencoder_path = (
                args.runs_dir
                / train_hash
                / "autoencoder"
                / f"seed_{seed}"
                / "model.pt"
            )

            out_dir = (
                args.runs_dir
                / train_hash
                / "probes"
                / "linear"
                / f"seed_{seed}"
            )

            classifier_path = out_dir / "classifier.pt"
            loss_path = out_dir / "loss.jsonl"
            metrics_path = out_dir / "metrics.json"

            if not autoencoder_path.exists():
                raise FileNotFoundError(
                    f"Missing autoencoder checkpoint for seed {seed}: {autoencoder_path}"
                )

            if (
                not args.overwrite
                and classifier_path.exists()
                and loss_path.exists()
                and metrics_path.exists()
            ):
                print(f"skip: {out_dir}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            if loss_path.exists():
                loss_path.unlink()
            if classifier_path.exists():
                classifier_path.unlink()
            if metrics_path.exists():
                metrics_path.unlink()

            autoencoder = AutoEncoder(run_config).to(device)

            checkpoint = torch.load(autoencoder_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
            else:
                autoencoder.load_state_dict(checkpoint)

            autoencoder.eval()

            for parameter in autoencoder.parameters():
                parameter.requires_grad_(False)

            classifier = nn.Linear(run_config.K, args.num_classes).to(device)

            optimizer = optim.AdamW(
                classifier.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            data_manager = DataManager(seed=seed, config=run_config)
            trainloader, testloader = data_manager.get_probe_loaders(
                batch_size=args.batch_size,
            )

            step = 0

            with loss_path.open("w", encoding="utf-8") as f:
                for epoch in range(args.epochs):
                    classifier.train()

                    for x, y in trainloader:
                        x = x.to(device)
                        y = y.to(device)

                        with torch.no_grad():
                            z = autoencoder.encode(x)

                        logits = classifier(z)
                        loss = F.cross_entropy(logits, y)

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                        predictions = logits.argmax(dim=1)
                        accuracy = (predictions == y).float().mean()

                        row = {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(loss.detach().cpu()),
                            "accuracy": float(accuracy.detach().cpu()),
                        }

                        f.write(json.dumps(row) + "\n")

                        print(
                            f"seed={seed} epoch={epoch} step={step} "
                            f"loss={row['loss']:.6g} acc={row['accuracy']:.4f}"
                        )

                        step += 1

            classifier.eval()

            total_correct = 0
            total_count = 0
            total_loss = 0.0

            with torch.no_grad():
                for x, y in testloader:
                    x = x.to(device)
                    y = y.to(device)

                    z = autoencoder.encode(x)
                    logits = classifier(z)
                    loss = F.cross_entropy(logits, y, reduction="sum")

                    predictions = logits.argmax(dim=1)

                    total_loss += float(loss.detach().cpu())
                    total_correct += int((predictions == y).sum().detach().cpu())
                    total_count += int(y.numel())

            metrics = {
                "test_loss": total_loss / total_count,
                "test_accuracy": total_correct / total_count,
                "num_test_examples": total_count,
            }

            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "run_config": asdict(run_config),
                    "seed": seed,
                    "autoencoder_path": str(autoencoder_path),
                    "probe": {
                        "type": "linear",
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "num_classes": args.num_classes,
                    },
                    "metrics": metrics,
                },
                classifier_path,
            )

            print(f"saved: {classifier_path}")
            print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
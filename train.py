import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import BCRNet
from loss import BCRNetLoss
from dataset import IRSTDDataset
from metrics import IRSTDMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train BCR-Net")
    parser.add_argument("--dataset-root", default="datasets/IRSTD-1K")
    parser.add_argument("--train-index", default="datasets/IRSTD-1K/img_idx/train_IRSTD-1K.txt")
    parser.add_argument("--mean", default=87.4661865234375, type=float)
    parser.add_argument("--std", default=39.71953201293945, type=float)
    parser.add_argument("--save-dir", default="runs/irstd1k")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--positive-probability", type=float, default=0.5)
    parser.add_argument("--curriculum-epochs", type=int, default=300)
    parser.add_argument("--alpha-end", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-index", default="datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--ignore-empty-for-niou", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_loss, best_miou=-1.0):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_miou": best_miou,
        },
        path,
    )


@torch.inference_mode()
def evaluate(model, loader, device, threshold, ignore_empty_for_niou):
    model.eval()
    metrics = IRSTDMetrics(
        threshold=threshold,
        ignore_empty_for_niou=ignore_empty_for_niou,
    )
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        probability = torch.sigmoid(model(image)[-1])
        metrics.update(probability, batch["mask"], batch["size"])
    return metrics.compute()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = IRSTDDataset(
        root=args.dataset_root,
        index_file=args.train_index,
        mean=args.mean,
        std=args.std,
        training=True,
        patch_size=args.patch_size,
        positive_probability=args.positive_probability,
        boundary_width=args.boundary_width,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = None
    if args.val_index:
        val_dataset = IRSTDDataset(
            root=args.dataset_root,
            index_file=args.val_index,
            mean=args.mean,
            std=args.std,
            training=False,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = BCRNet().to(device)
    criterion = BCRNetLoss(
        alpha_end=args.alpha_end,
        curriculum_epochs=args.curriculum_epochs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - min(step / total_steps, 1.0)) ** 0.9,
    )

    start_epoch = 1
    best_loss = float("inf")
    best_miou = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_miou = float(checkpoint.get("best_miou", best_miou))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss, _ = criterion(outputs, target, boundary, epoch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())

        epoch_loss = running_loss / max(1, len(loader))
        print(f"epoch={epoch:04d} loss={epoch_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        save_checkpoint(save_dir / "bcrnet_latest.pt", epoch, model, optimizer, scheduler, best_loss, best_miou)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(save_dir / "bcrnet_best_loss.pt", epoch, model, optimizer, scheduler, best_loss, best_miou)
        if val_loader is not None and (epoch % max(1, args.val_interval) == 0 or epoch == args.epochs):
            result = evaluate(
                model,
                val_loader,
                device,
                args.val_threshold,
                args.ignore_empty_for_niou,
            )
            print(
                "validation="
                + " ".join(f"{name}={value:.6f}" for name, value in result.items()),
                flush=True,
            )
            if result["mIoU"] > best_miou:
                best_miou = result["mIoU"]
                save_checkpoint(save_dir / "bcrnet_best_miou.pt", epoch, model, optimizer, scheduler, best_loss, best_miou)


if __name__ == "__main__":
    main()

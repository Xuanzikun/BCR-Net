import argparse

import torch
from torch.utils.data import DataLoader

from model import BCRNet
from checkpoints import load_checkpoint
from dataset import IRSTDDataset
from metrics import IRSTDMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BCR-Net")
    parser.add_argument("--dataset-root", default="datasets/IRSTD-1K")
    parser.add_argument("--test-index", default="datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt")
    parser.add_argument("--checkpoint", default="runs/irstd1k/bcrnet_best_miou.pt")
    parser.add_argument("--mean", default=87.4661865234375, type=float)
    parser.add_argument("--std", default=39.71953201293945, type=float)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ignore-empty-for-niou", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = IRSTDDataset(
        root=args.dataset_root,
        index_file=args.test_index,
        mean=args.mean,
        std=args.std,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = BCRNet().to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    metrics = IRSTDMetrics(
        threshold=args.threshold,
        ignore_empty_for_niou=args.ignore_empty_for_niou,
    )
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            probability = torch.sigmoid(model(image)[-1])
            metrics.update(probability, batch["mask"], batch["size"])
    result = metrics.compute()
    for name, value in result.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()

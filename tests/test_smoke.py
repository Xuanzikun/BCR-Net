import torch

from model import BCRNet
from loss import BCRNetLoss
from dataset import generate_boundary_label
from metrics import IRSTDMetrics


def test_forward_and_loss():
    model = BCRNet()
    image = torch.randn(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 24:32, 25:33] = 1
    boundary = torch.from_numpy(generate_boundary_label(target[0, 0], width=2))[None, None]
    outputs = model(image)
    assert len(outputs) == 4
    assert all(output.shape == target.shape for output in outputs)
    loss, details = BCRNetLoss()(outputs, target, boundary, epoch=1)
    assert torch.isfinite(loss)
    assert details["alpha"] == 1.0


def test_fa_counts_unmatched_region_with_same_area_as_match():
    prediction = torch.zeros(1, 1, 10, 10)
    target = torch.zeros(1, 1, 10, 10)
    prediction[0, 0, 1, 1] = 1
    target[0, 0, 1, 1] = 1
    prediction[0, 0, 8, 8] = 1

    metrics = IRSTDMetrics(threshold=0.5)
    metrics.update(prediction, target)
    result = metrics.compute()

    assert metrics.false_alarm_pixels == 1
    assert result["Fa"] == 0.01

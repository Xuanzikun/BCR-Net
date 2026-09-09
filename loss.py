import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    """Sample-wise soft IoU loss operating on logits."""

    def forward(self, logits, target):
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum(dim=(1, 2, 3))
        union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        return 1.0 - ((intersection + 1.0) / (union + 1.0)).mean()


class BCRNetLoss(nn.Module):
    """Training objective used for the full BCR-Net model."""

    def __init__(
        self,
        lambda_body=0.5,
        lambda_boundary=0.5,
        lambda_compensation=0.1,
        positive_weight=20.0,
        alpha_end=0.25,
        curriculum_epochs=300,
    ):
        super().__init__()
        self.lambda_body = float(lambda_body)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_compensation = float(lambda_compensation)
        self.alpha_end = float(alpha_end)
        self.curriculum_epochs = int(curriculum_epochs)
        self.soft_iou = SoftIoULoss()
        self.register_buffer("positive_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def alpha(self, epoch):
        total = max(1, self.curriculum_epochs)
        progress = min(max((int(epoch) - 1) / float(total), 0.0), 1.0)
        return 1.0 + (self.alpha_end - 1.0) * progress

    def segmentation_loss(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.positive_weight.to(device=logits.device, dtype=logits.dtype),
        )
        return self.soft_iou(logits, target) + bce

    def forward(self, outputs, target, boundary, epoch):
        if len(outputs) != 4:
            raise ValueError("BCR-Net must return body, boundary, compensation, and final logits")
        body_logits, boundary_logits, compensation_logits, final_logits = outputs

        loss_final = self.segmentation_loss(final_logits, target)
        loss_body = self.segmentation_loss(body_logits, target)
        loss_boundary = self.segmentation_loss(boundary_logits, boundary)

        with torch.no_grad():
            final_probability = torch.sigmoid(final_logits.detach())
            residual_target = torch.clamp(target - final_probability, min=0.0, max=1.0)
            alpha = self.alpha(epoch)
            compensation_target = alpha * target + (1.0 - alpha) * residual_target

        loss_compensation = nn.functional.binary_cross_entropy_with_logits(
            compensation_logits,
            compensation_target,
            pos_weight=self.positive_weight.to(
                device=compensation_logits.device,
                dtype=compensation_logits.dtype,
            ),
        )
        total = (
            loss_final
            + self.lambda_body * loss_body
            + self.lambda_boundary * loss_boundary
            + self.lambda_compensation * loss_compensation
        )
        details = {
            "total": total.detach(),
            "final": loss_final.detach(),
            "body": loss_body.detach(),
            "boundary": loss_boundary.detach(),
            "compensation": loss_compensation.detach(),
            "alpha": alpha,
        }
        return total, details

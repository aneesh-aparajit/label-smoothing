import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, eps: float = 0.1, reduction: str = None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = F.log_softmax(y_pred, dim=-1)
        y_true = y_true.to(dtype=torch.long)

        C = y_pred.shape[-1]

        loss = None

        log_preds = -y_pred

        if self.reduction == None:
            raise Exception("Reduction cannot be None")
        elif self.reduction == "sum":
            log_preds = log_preds.sum()
        elif self.reduction == "mean":
            '''We find the sum across dimension -1, because we also divide it by class in the next step, so, 
            if we take mean here itself, we will be reducing the impact of this term in the final loss function, 
            so, we sum it across the last dimension and then, we take mean, so we end up dividing it only by C and not C^2.
            '''
            log_preds = log_preds.sum(dim=-1).mean()

        loss = (self.eps / C) * log_preds

        loss += (1 - self.eps) * F.nll_loss(input=y_pred, target=y_true, reduction=self.reduction)

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    y_int = True

    def __init__(self, eps: float = 0.1, reduction="mean"):
        self.eps, self.reduction = eps, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(
                dim=-1
            )  # We divide by that size at the return line so sum and not mean
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target.long(), reduction=self.reduction
        )

    def activation(self, out):
        return F.softmax(out, dim=-1)

    def decodes(self, out):
        return out.argmax(dim=-1)


if __name__ == "__main__":
    custom_loss_fct = LabelSmoothingCrossEntropyLoss(eps=0.1, reduction="mean")
    fastai_loss_fct = LabelSmoothingCrossEntropy(eps=0.1, reduction="mean")

    y_pred = torch.randn(size=(5, 5))
    y_true = torch.tensor([1, 2, 3, 3, 3], dtype=torch.long)

    print(f"Custom Loss : {custom_loss_fct.forward(y_pred=y_pred, y_true=y_true)}")
    print(
        f'PyTorch Loss: {F.cross_entropy(input=y_pred, target=y_true, label_smoothing=0.1, reduction="mean")}'
    )
    print(f"Fast.ai Loss: {fastai_loss_fct.forward(output=y_pred, target=y_true)}")

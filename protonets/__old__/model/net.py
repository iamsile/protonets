import torch
import torch.nn as nn
from torch.nn import functional as F


class Protonet(nn.Module):
    """Prototypical network architecture."""

    def __init__(self, dim_in: int = 1, dim_hid: int = 64, dim_z: int = 64):
        super(Protonet, self).__init__()
        self.embedding = nn.Sequential(
            self._conv_block(dim_in, dim_hid),
            self._conv_block(dim_hid, dim_hid),
            self._conv_block(dim_hid, dim_hid),
            self._conv_block(dim_hid, dim_z)
        )

    def forward(self, x: torch.Tensor):
        z = self.embedding(x)
        return z.view(z.size(0), -1)  # flatten

    def criterion(self, y_hat, y, num_support):
        classes = torch.unique(y)
        num_classes = len(classes)
        num_query = y.eq(classes[0].item()).sum().item() - num_support
        support_idxs = [y.eq(c).nonzero()[:num_support].squeeze(1)
                        for c in classes]
        prototypes = torch.stack([y_hat[idx_list].mean(0)
                                  for idx_list in support_idxs])
        query_idxs = torch.stack(
            [y.eq(c).nonzero()[num_support:] for c in classes]).view(-1)
        query_samples = y_hat[query_idxs]
        dists = self._euclidean_distance(query_samples, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)
        target_inds = torch.arange(0, num_classes).view(
            num_classes, 1, 1).expand(num_classes, num_query, 1).long()
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
        return loss_val, acc_val

    def _euclidean_distance(self, x, y):
        """Euclidean distance between two tensors,
        `x`, `y`, such that:
        `x: N x D` and `y: M x D`
        """
        N = x.size(0)
        M = y.size(0)
        D = x.size(1)
        assert D == y.size(1)
        x = x.unsqueeze(1).expand(N, M, D)
        y = x.unsqueeze(0).expand(N, M, D)
        return torch.pow(x-y, 2).sum(2)

    def _conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

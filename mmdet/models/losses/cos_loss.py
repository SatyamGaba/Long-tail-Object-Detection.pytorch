import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Module):
    """Centsr loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, in_features, out_features, eps=1e-7, s=30.0, m=0.4, use_gpu = True):
        super(CosLoss, self).__init__()
        self.s = s #30.0
        self.m = m #0.4
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features, bias=False)
        self.eps = None
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.fc = self.fc.cuda()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        size = 0
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
            size+=1

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
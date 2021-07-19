import torch
class FocalLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]
    ce loss:
    loss = -log(p')
    focal loss:
    loss = -alpha*((1-p')^gamma)*(log(p'))
    Args:
        size (int): the number of class
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        alpha: the param for focal loss
        gamma: the param for focal loss
    """
    def __init__(
        self,
        size: int,
        smoothing: float,
        alpha: int=0.25,
        gamma: int=2
    ):
        """Construct an LabelSmoothingLoss object."""
        super(FocalLoss, self).__init__()
        self.size = size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(1) == self.size
        batch_size = x.size(0)
        targets = targets.view(-1,1)
        probs = torch.softmax(x, dim=1)
        true_dist = torch.zeros_like(x).to(x.device)
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, targets, self.confidence)
        log_p = probs.log()
        ce_loss = log_p * true_dist
        alpha = true_dist*(2*self.alpha-1) + (1-self.alpha)
        # 0 -> 1-a
        # 1 -> a
        loss = -alpha * torch.pow(1-probs, self.gamma) * ce_loss
        loss = loss.sum(1)/batch_size
        return loss

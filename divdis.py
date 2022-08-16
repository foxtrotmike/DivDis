import torch
from einops import rearrange
from torch import nn


def to_probs(logits, heads):
    """
    Converts logits to probabilities.
    Input must have shape [batch_size, heads * classes].
    Output will have shape [batch_size, heads, classes].
    """

    B, N = logits.shape
    if N == heads:  # Binary classification; each head outputs a single scalar.
        preds = logits.sigmoid().unsqueeze(-1)
        probs = torch.cat([preds, 1 - preds], dim=-1)
    else:
        logits_chunked = torch.chunk(logits, heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=1).softmax(-1)
    B, H, D = probs.shape
    assert H == heads
    return probs


def get_disagreement_scores(logits, heads, mode="l1"):
    probs = to_probs(logits, heads)
    if mode == "l1":  # This was used in the paper
        diff = probs.unsqueeze(1) - probs.unsqueeze(2)
        disagreement = diff.abs().mean([-3, -2, -1])
    elif mode == "kl":
        marginal_p = probs.mean(dim=0)  # H, D
        marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> 1 (h g) (d e)")  # 1, H^2, D^2

        pointwise_p = torch.einsum("bhd,bge->bhgde", probs, probs)  # B, H, H, D, D
        pointwise_p = rearrange(
            pointwise_p, "b h g d e -> b (h g) (d e)"
        )  # B, H^2, D^2

        kl_computed = pointwise_p * (pointwise_p.log() - marginal_p.log())
        kl_grid = rearrange(kl_computed.sum(-1), "b (h g) -> b h g", h=heads)
        disagreement = torch.triu(kl_grid, diagonal=1).sum([-1, -2])
    return disagreement.argsort(descending=True)


class DivDisLoss(nn.Module):
    """Computes pairwise repulsion losses for DivDis.

    Args:
        logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * DIM].
        heads (int): Number of heads.
        mode (str): DIVE loss mode. One of {pair_mi, total_correlation, pair_l1}.
    """

    def __init__(self, heads, mode="mi", reduction="mean"):
        super().__init__()
        self.heads = heads
        self.mode = mode
        self.reduction = reduction

    def forward(self, logits):
        heads, mode, reduction = self.heads, self.mode, self.reduction
        probs = to_probs(logits, heads)
        #Batch x Heads x 2 (for classes 1,0)

        if mode == "mi":  # This was used in the paper
            #import pdb;pdb.set_trace()
            marginal_p = probs.mean(dim=0)  # H, D
            """
            H = numner of heads, D = number of probability dimensions (2 for classification)
            m(h,d) = marginal probability of generating label d by head h
            tensor([[0.6390, 0.3610],
                    [0.5032, 0.4968]], grad_fn=<MeanBackward1>)
            """
            marginal_p = torch.einsum(
                "hd,ge->hgde", marginal_p, marginal_p
            )  # H, H, D, D
            """
            for the average probability acorss all examples
            h,g = heads, d,e = probbaility dimensions (2 for 0,1 class)
            m(h,g,d,e) <- m(h,d)*m(g,e) #m(0,0,0,0) = m(h = 0,d=0)*m(g=0, e = 0) = 0.639*0.639  = 0.4083, #m(0,0,0,1) = m(h = 0,d=0)*m(g=0, e = 1)=0.639*0.361 = 0.2307
            tensor([[[[0.4083, 0.2307],
                      [0.2307, 0.1303]],
                                     #m(0,1,0,0) = m(h = 0,d=0)*m(g=1, e = 0)=0.639*0.5032 = 0.3215
                     [[0.3215, 0.3174],
                      [0.1817, 0.1794]]],
            
            
                    [[[0.3215, 0.1817],
                      [0.3174, 0.1794]],
            
                     [[0.2532, 0.2500],
                      [0.2500, 0.2468]]]], grad_fn=<MulBackward0>)
            """
            marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2
            """
            #flatten out the last dimension -- descriptor of p(h) x (g)
            m(h+H*g,d+D*e = m(h,g,d,e) #
            tensor([[0.4083, 0.2307, 0.2307, 0.1303],
                    [0.3215, 0.3174, 0.1817, 0.1794],
                    [0.3215, 0.1817, 0.3174, 0.1794],
                    [0.2532, 0.2500, 0.2500, 0.2468]], grad_fn=<ReshapeAliasBackward0>)
            """

            joint_p = torch.einsum("bhd,bge->bhgde", probs, probs)
            """
            for each example, compute j(b,h,g,d,e) = p(b,h,d)*p(b,g,e)
            for example for exmple b = 0 with probs[0]:
                tensor([[1.3292e-02, 9.8671e-01],
                        [1.0000e+00, 2.3842e-07]], grad_fn=<SelectBackward0>)
                
            we get joint_p[0] as follows which is a descriptor of p(h,g) for example b = 0
                tensor([[[[1.7669e-04, 1.3116e-02],
                          [1.3116e-02, 9.7359e-01]],
                
                         [[1.3292e-02, 3.1692e-09],
                          [9.8671e-01, 2.3525e-07]]],
                
                
                        [[[1.3292e-02, 9.8671e-01],
                          [3.1692e-09, 2.3525e-07]],
                
                         [[1.0000e+00, 2.3842e-07],
                          [2.3842e-07, 5.6843e-14]]]], grad_fn=<SelectBackward0>)
            """
            joint_p = joint_p.mean(dim = 0) # H, H, D, D
            """
            simple average across all examples to get j(h,g,d,e)
            descriptor of joint probability p(h,g)
                tensor([[[[0.6186, 0.0204],
                          [0.0204, 0.3406]],
                
                         [[0.3377, 0.3013],
                          [0.1655, 0.1955]]],
                
                
                        [[[0.3377, 0.1655],
                          [0.3013, 0.1955]],
                
                         [[0.4853, 0.0179],
                          [0.0179, 0.4789]]]], grad_fn=<MeanBackward1>)
            """
            joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2
            """
             descriptor of joint probability p(h,g)
             j(h+H*g,d+D*e) = j(h,g,d,e) #
             
             tensor([[0.6186, 0.0204, 0.0204, 0.3406],
                    [0.3377, 0.3013, 0.1655, 0.1955],
                    [0.3377, 0.1655, 0.3013, 0.1955],
                    [0.4853, 0.0179, 0.0179, 0.4789]],
            """

            # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
            # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
            kl_computed = joint_p * (joint_p.log() - marginal_p.log())
            """
            j(h+H*g,d+D*e)*log(j(h+H*g,d+D*e)/m(h+H*g,d+D*e))
            
            tensor([[ 0.2570, -0.0495, -0.0495,  0.3272],
                    [ 0.0166, -0.0158, -0.0154,  0.0169],
                    [ 0.0166, -0.0154, -0.0158,  0.0169],
                    [ 0.3157, -0.0472, -0.0472,  0.3174]], grad_fn=<MulBackward0>)
            """
            kl_computed = kl_computed.sum(dim=-1)
            """
            KL(h+H*g) = sum_(d,e): j(h+H*g,d+D*e)*log(j(h+H*g,d+D*e)/m(h+H*g,d+D*e))
                tensor([0.4852, 0.0023, 0.0023, 0.5387], grad_fn=<SumBackward1>)
            """
            kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
            """
            KL(h,g) = KL(h+H*g)
                tensor([[0.4852, 0.0023],
                        [0.0023, 0.5387]], grad_fn=<ReshapeAliasBackward0>)
            """
            repulsion_grid = -kl_grid
        elif mode == "l1":
            dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
            dists = dists.sum(dim=-1).mean(dim=0)
            repulsion_grid = dists
        else:
            raise ValueError()#f"{mode=} not implemented!"

        if reduction == "mean":  # This was used in the paper
            repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
            """
            tensor([[ 0.0000, -0.0023],
                    [ 0.0000,  0.0000]], grad_fn=<TriuBackward0>)
            """
            repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
            """
            tensor([-0.0023], grad_fn=<IndexBackward0>)
            """
            repulsion_loss = -repulsions.mean()
            """
            tensor(0.0023, grad_fn=<NegBackward0>)
            """
            
        elif reduction == "min_each":
            repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
                repulsion_grid, diagonal=-1
            )
            rows = [r for r in repulsion_grid]
            row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
            repulsion_loss = -torch.stack(row_mins).mean()
        else:
            raise ValueError()#f"{reduction=} not implemented!"

        return repulsion_loss

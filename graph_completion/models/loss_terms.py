"""
Module containing the implementation of the specific loss functions utilized by embedding models from related work.
"""

import attr
import torch as pt
from attr.validators import in_, instance_of
from torch import nn
from torch.nn.functional import logsigmoid, relu, softmax

from graph_completion.models.link_rankers import LinkPredictorKBGAT, LinkPredictorMLP, LinkPredictorSACN, \
    distance_Context, distance_RotatE, distance_TransE, score_ComplEx, score_DistMult
from graph_completion.utils import AbstractConf


class EmbeddingLossMLP(nn.Module):
    """
    Class implementing the MLP loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_emb_neg: pt.Tensor, link_ranker: LinkPredictorMLP):
        loss_pos_term = -logsigmoid(link_ranker(edge_emb))
        num_negative_samples, embedding_dim = edge_emb_neg.size(2), edge_emb_neg.size(3)
        edge_emb_neg = edge_emb_neg.view(2, -1, embedding_dim)
        edge_emb_neg[1] *= -1
        loss_neg_term = pt.mean(-logsigmoid(link_ranker(edge_emb_neg)).view(-1, num_negative_samples), dim=1)
        loss = pt.mean(loss_pos_term + loss_neg_term)
        return loss


class EmbeddingLossTransE(nn.Module):
    """
    Class implementing the TransE loss.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor, edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor):
        loss_pos_term = -logsigmoid(self.margin - distance_TransE(edge_emb[0], edge_attr, edge_emb[1]))
        loss_neg_term = pt.mean(-logsigmoid(
            distance_TransE(edge_emb_neg[0], edge_attr_neg, edge_emb_neg[1]) - self.margin
        ), dim=1)
        loss = pt.mean(0.5 * (loss_pos_term + loss_neg_term))
        return loss


class EmbeddingLossDistMult(nn.Module):
    """
    Class implementing the DistMult loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor, edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor):
        loss_pos_term = -logsigmoid(score_DistMult(edge_emb[0], edge_attr, edge_emb[1]))
        loss_neg_term = pt.mean(-logsigmoid(
            -score_DistMult(edge_emb_neg[0], edge_attr_neg, edge_emb_neg[1])
        ), dim=1)
        loss = pt.mean(0.5 * (loss_pos_term + loss_neg_term))
        return loss


class EmbeddingLossComplEx(nn.Module):
    """
    Class implementing the ComplEx loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor, edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor):
        loss_pos_term = -logsigmoid(score_ComplEx(edge_emb[0], edge_attr, edge_emb[1]))
        loss_neg_term = pt.mean(-logsigmoid(
            -score_ComplEx(edge_emb_neg[0], edge_attr_neg, edge_emb_neg[1])
        ), dim=1)
        loss = pt.mean(0.5 * (loss_pos_term + loss_neg_term))
        return loss


class EmbeddingLossRotatE(nn.Module):
    """
    Class implementing the RotatE loss.
    """

    def __init__(self, margin: float, rotate_alpha: float):
        super().__init__()
        self.margin = margin
        self.alpha = rotate_alpha

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor, edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor):
        loss_pos_term = -logsigmoid(self.margin - distance_RotatE(edge_emb[0], edge_attr, edge_emb[1]))
        dist_neg_terms = distance_RotatE(edge_emb_neg[0], edge_attr_neg, edge_emb_neg[1])
        loss_neg_terms = -logsigmoid(dist_neg_terms - self.margin)
        weight_neg_terms = softmax(-dist_neg_terms.detach() * self.alpha, dim=1)
        loss_neg_term = pt.sum(weight_neg_terms * loss_neg_terms, dim=1)
        loss = pt.mean(0.5 * (loss_pos_term + loss_neg_term))
        return loss


class EmbeddingLossGATNE(nn.Module):
    """
    Class implementing the GATNE loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_emb_neg: pt.Tensor):
        loss_pos_term = -logsigmoid(-distance_Context(edge_emb[0], edge_emb[1]))
        loss_neg_term = pt.mean(-logsigmoid(distance_Context(edge_emb_neg[0], edge_emb_neg[1])), dim=1)
        loss = pt.mean(0.5 * (loss_pos_term + loss_neg_term))
        return loss


class EmbeddingLossSACN(nn.Module):
    """
    Class implementing the SACN loss.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="sum")

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor,
                edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor, link_ranker: LinkPredictorSACN):
        batch_size, device = edge_emb.size(1), edge_emb.device
        loss_pos_term = self.bce(link_ranker(edge_emb, edge_attr), pt.ones(batch_size, device=device))
        num_negative_samples, embedding_dim = edge_emb_neg.size(2), edge_emb_neg.size(3)
        edge_emb_neg = edge_emb_neg.view(2, -1, embedding_dim)
        edge_attr_neg = edge_attr_neg.view(-1, embedding_dim)
        loss_neg_term = self.bce(link_ranker(edge_emb_neg, edge_attr_neg),
                                 pt.zeros(batch_size * num_negative_samples, device=device))
        loss = (loss_pos_term + loss_neg_term) / (batch_size * (1 + num_negative_samples))
        return loss


class EmbeddingLossKBGAT(nn.Module):
    """
    Class implementing the KBGAT loss.
    """

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin
        self.soft_margin = nn.SoftMarginLoss(reduction="sum")

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor,
                edge_emb_neg: pt.Tensor, edge_attr_neg: pt.Tensor, link_ranker: LinkPredictorKBGAT):
        batch_size, device = edge_emb.size(1), edge_emb.device
        loss_emb_pos_term = distance_TransE(edge_emb[0], edge_attr, edge_emb[1], p=1)
        loss_emb_neg_term = pt.mean(distance_TransE(edge_emb_neg[0], edge_attr_neg, edge_emb_neg[1], p=1), dim=1)
        loss_embedder = pt.mean(relu(self.margin + loss_emb_pos_term - loss_emb_neg_term))

        loss_ranker_pos_term = self.soft_margin(link_ranker(edge_emb, edge_attr), pt.ones(batch_size, device=device))
        num_negative_samples, embedding_dim = edge_emb_neg.size(2), edge_emb_neg.size(3)
        edge_emb_neg = edge_emb_neg.view(2, -1, embedding_dim)
        edge_attr_neg = edge_attr_neg.view(-1, embedding_dim)
        loss_ranker_neg_term = self.soft_margin(link_ranker(edge_emb_neg, edge_attr_neg),
                                                -pt.ones(batch_size * num_negative_samples, device=device))
        loss_ranker = (loss_ranker_pos_term + loss_ranker_neg_term) / (batch_size * (1 + num_negative_samples))
        loss = 0.5 * (loss_embedder + loss_ranker)
        return loss


@attr.s
class EmbeddingLossHpars(AbstractConf):
    OPTIONS = {"mlp": EmbeddingLossMLP, "transe": EmbeddingLossTransE, "distmult": EmbeddingLossDistMult,
               "complex": EmbeddingLossComplEx, "rotate": EmbeddingLossRotatE,
               "gatne": EmbeddingLossGATNE, "sacn": EmbeddingLossSACN, "kbgat": EmbeddingLossKBGAT}
    algorithm = attr.ib(validator=in_(list(OPTIONS.keys())))
    margin = attr.ib(default=1.0, validator=instance_of(float))
    rotate_alpha = attr.ib(default=1.0, validator=instance_of(float))

    def __attrs_post_init__(self):
        self.name = self.algorithm

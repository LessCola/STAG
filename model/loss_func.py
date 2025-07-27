import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def mse_loss(x, y):
    loss = (x - y).pow_(2)
    loss = loss.mean()
    return loss


def sample_negatives(adjacency_matrix, N, valid_nodes=None):
    """
    random_scores is a matrix of random values.
    Self-connections are masked by setting their scores to -1.
    torch.topk is used to select the indices with the highest scores for each node.
    """

    num_nodes = adjacency_matrix.shape[0]
    device = adjacency_matrix.device

    if N == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    if N == np.inf:
        diag_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)
        return ~diag_mask

    # * graph classification, some graphs contain no masked nodes or only one node
    if num_nodes == 0 or num_nodes == 1:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    num_negatives = min(num_nodes - 1, N if N >= 1 else math.ceil((num_nodes - 1) * N))

    # Assign random scores to each node pair
    random_scores = torch.rand((num_nodes, num_nodes), device=device)

    # Mask out self-connections by setting their scores to a very low value
    random_scores[torch.arange(num_nodes), torch.arange(num_nodes)] = -1
    # ! Mask out invalid nodes by setting their scores to a very low value
    if valid_nodes is not None:
        invalid_nodes = torch.ones(num_nodes, dtype=torch.bool, device=device)
        invalid_nodes[valid_nodes] = False
        random_scores[:, invalid_nodes] = -1
        random_scores[invalid_nodes, :] = -1

    # Select the indices with the highest scores, excluding self-connections
    _, neg_indices = torch.topk(random_scores, num_negatives, dim=1)

    # Create the negative mask
    neg_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    row_indices = torch.arange(num_nodes).unsqueeze(1).expand_as(neg_indices)
    neg_mask[row_indices, neg_indices] = True

    return neg_mask


def exp_temp_stabilized(similarity, tau, max_clip_value=20):
    # Scale the similarity
    scaled_sim = similarity / (tau + 1e-10)
    # Clamp the scaled similarity to avoid too large values
    scaled_sim = torch.clamp(scaled_sim, min=-max_clip_value, max=max_clip_value)
    # Apply the exponential function
    exp_sim = torch.exp(scaled_sim)
    # Handle extreme values post-exponentiation
    exp_sim = torch.clamp(exp_sim, max=np.exp(max_clip_value))
    return exp_sim


def compute_negative_similarities(z_graph, neg_mask, tau):
    rows, cols = torch.where(neg_mask)
    # Compute the similarities only for these masked elements
    sim_values = (z_graph[rows] * z_graph[cols]).sum(dim=1)
    # Apply temperature scaling and exponentiation (exp_temp_stabilized)
    sim_values = exp_temp_stabilized(sim_values, tau)
    # Create an empty similarity matrix and place the computed similarities
    sim_matrix = torch.zeros_like(neg_mask, dtype=torch.float)
    sim_matrix[rows, cols] = sim_values
    # Sum the similarities along the rows to get the negative similarities
    neg_sim = sim_matrix.sum(dim=1)
    return neg_sim


class ContrastiveLoss(nn.Module):
    """compute intra-view loss for the masked view and allow the negative samples to be drawn from other graphs in the same batch"""

    def __init__(
        self,
        tau=0.5,
        N=np.inf,
        epsilon=1.0e-5,
    ):
        super().__init__()
        self.tau = tau
        self.N = np.inf if N == -1 else N
        self.epsilon = epsilon

    def forward(self, g, x, y, mask_nodes=None):
        # g is the batched graphs
        # Normalize the embeddings
        if mask_nodes is None:
            z1 = F.normalize(x, p=2, dim=-1)
            z2 = F.normalize(y, p=2, dim=-1)
        else:
            z1 = F.normalize(x[mask_nodes], p=2, dim=-1)
            z2 = F.normalize(y[mask_nodes], p=2, dim=-1)

        if mask_nodes is not None:
            adjacency_matrix = g.adj(etype=("_N", "_E", "_N")).to_dense()[mask_nodes][
                :, mask_nodes
            ]
        else:
            adjacency_matrix = g.adj(etype=("_N", "_E", "_N")).to_dense()

        # draw negative samples from other graphs in the same batch
        pos_sim = exp_temp_stabilized(torch.sum(z1 * z2, dim=1), self.tau)
        if self.N != 0:
            neg_mask = sample_negatives(
                adjacency_matrix,
                self.N,
            )

            neg_sim = compute_negative_similarities(z1, neg_mask, self.tau)

            # Calculate loss for the current graph
            loss = -torch.log(pos_sim / (pos_sim + neg_sim + self.epsilon))
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        else:
            loss = -torch.log(pos_sim)
        return loss.mean()


class CosineCommitLoss(nn.Module):
    def __init__(self, commit_score=1):
        """
        Initializes the CommitLoss module.

        Parameters:
            commit_score (float): The weight for the commitment loss term.
        """
        super(CosineCommitLoss, self).__init__()
        self.commit_score = commit_score

    def forward(self, z_e, z_q):
        """
        Forward pass for calculating the commitment loss using cosine similarity
        with soft assignments.

        Parameters:
            z_e (torch.Tensor): The encoder outputs (before quantization), shape [n, d].
            z_q (torch.Tensor): The weighted combination of codebook vectors, shape [n, d].

        Returns:
            loss (torch.Tensor): The weighted commitment loss.
        """
        # Normalize z_e and z_q
        z_e_norm = F.normalize(z_e, dim=1)
        z_q_norm = F.normalize(z_q, dim=1)

        # Compute cosine similarity directly with weighted combination
        cosine_similarity = torch.sum(z_e_norm * z_q_norm.detach(), dim=1)

        # Convert to cosine distance
        cosine_distance = 1 - cosine_similarity

        # Average across batch
        loss = self.commit_score * torch.mean(cosine_distance)

        return loss

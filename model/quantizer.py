import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

from .loss_func import CosineCommitLoss as CommitLoss


class VQ(nn.Module):
    def __init__(
        self,
        args,
        top_k,
        raw_temp=0.02,
        quant_temp=0.05,
        commit_score=0.25,
    ):
        super(VQ, self).__init__()
        self.args = args
        self.top_k = top_k

        checkpoint = torch.load(
            "codebook/subword_embeddings.pth",
            map_location="cpu",
        )
        self.codebook = np.load(
            "codebook/subword_vocabulary.npy",
            allow_pickle=True,
        )

        self.num_tokens = checkpoint.shape[0]
        self.embed_dim = checkpoint.shape[1]
        print("Codebook Size: {}".format(self.num_tokens))
        print("Feature Dim: {}".format(self.embed_dim))
        self.tok_embeddings = Embedding(self.num_tokens, self.embed_dim)
        self.tok_embeddings.weight.data = checkpoint
        self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.float()
        self.tok_embeddings.weight.requires_grad = (
            False  # Commented out to allow updates
        )
        # loss
        self.commit_score = commit_score
        self.commit_loss = CommitLoss(self.commit_score)

        # * temperature annealing
        # Temperature annealing parameters
        self.raw_temp = raw_temp
        self.quant_temp = max(self.raw_temp, quant_temp)

    def search_torch_soft(self, z_e):
        """
        Compute soft assignments with configurable sparsification behavior.

        Args:
            z_e: Input embeddings
            mode: Training mode - 'annealing', 'full', or 'smooth'
            inference: Whether in inference mode
        """

        # Project and normalize
        z_flattened = z_e.view(z_e.shape[0], -1)
        tok_embeddings_weight = self.tok_embeddings.weight

        z_flattened_norm = z_flattened / z_flattened.norm(dim=1, keepdim=True)
        tok_embeddings_weight_norm = tok_embeddings_weight / tok_embeddings_weight.norm(
            dim=1, keepdim=True
        )

        # Compute base attention weights
        cosine_similarity = torch.einsum(
            "bd,nd->bn", z_flattened_norm, tok_embeddings_weight_norm
        )
        logits = cosine_similarity / self.quant_temp
        attention_weights = F.softmax(logits, dim=-1)

        return attention_weights, tok_embeddings_weight

    def get_raw_attn(self, x):
        x_flattened = x.view(x.shape[0], -1)
        tok_embeddings_weight = self.tok_embeddings.weight

        x_flattened_norm = x_flattened / x_flattened.norm(dim=1, keepdim=True)
        tok_embeddings_weight_norm = tok_embeddings_weight / tok_embeddings_weight.norm(
            dim=1, keepdim=True
        )

        # Compute base attention weights
        cosine_similarity = torch.einsum(
            "bd,nd->bn", x_flattened_norm, tok_embeddings_weight_norm
        )
        logits = cosine_similarity / self.raw_temp
        attention_weights = F.softmax(logits, dim=-1)
        return attention_weights

    def quantize_torch(self, z_e, inference=False):
        return self.forward(z_e, inference)

    def forward(self, z_e, inference=False, raw_feat=None):
        if inference:
            # Use hard top-k selection during inference
            attention_weights, tok_embeddings_weight = self.search_torch_soft(z_e)
            z_q = torch.matmul(attention_weights, tok_embeddings_weight)
            _, top_indices = torch.topk(attention_weights, self.top_k, dim=1)
            tokens = self.get_token(top_indices)
            return z_q, tokens

        # Get soft assignments with temperature-dependent sparsity
        attention_weights, tok_embeddings_weight = self.search_torch_soft(z_e)

        # Compute weighted combination of codebook vectors
        z_q = torch.matmul(attention_weights, tok_embeddings_weight)

        # Commitment loss
        loss = self.commit_loss(z_e, z_q)

        # * alignment loss with raw quantized feature
        if raw_feat is not None:  # x is raw feature
            raw_attn = self.get_raw_attn(raw_feat)
            kl_loss = F.kl_div(
                (attention_weights + 1e-10).log(),
                raw_attn.detach(),
                reduction="batchmean",
                log_target=False,
            )
            loss += kl_loss * getattr(self.args, "kl_loss_scale", 1.0)
        # Straight-through gradient estimator
        z_q = z_e + (z_q - z_e).detach()

        return loss, z_q, None

    def get_token(self, min_indices):
        if isinstance(min_indices, torch.Tensor):
            min_indices = min_indices.cpu().numpy()
        texts = []
        for indices in min_indices:
            texts.append(self.codebook[indices])
        return texts

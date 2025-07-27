import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionModule(nn.Module):
    def __init__(self, d, d_prime, proj_hidden_dim=128):
        super(ProjectionModule, self).__init__()
        self.project_fX = nn.Sequential(
            nn.Linear(d_prime, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, d),
        )

    def __init_weights(self):
        # Initialize weights of the projection layer
        for module in self.project_fX:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X, f_X):
        # Project f(X) to dimension d
        f_X_proj = self.project_fX(f_X)
        return f_X_proj


class WeightedFusion(nn.Module):
    def __init__(self, d, d_prime, lean_x=0.6):
        """
        Args:
            d: dimension of X
            d_prime: dimension of f_X
            lean_x: weight for X (between 0 and 1), default 0.5 for true bisector
        """
        super(WeightedFusion, self).__init__()
        self.project_fX = nn.Linear(d_prime, d)

        # Convert lean_x to weights that sum to 1
        self.weight_params = nn.Parameter(torch.tensor([lean_x, 1.0 - lean_x]))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.project_fX.weight)
        if self.project_fX.bias is not None:
            nn.init.zeros_(self.project_fX.bias)

    def forward(self, X, f_X):
        # Project f_X to match dimension
        f_X_proj = self.project_fX(f_X)

        # Normalize vectors
        X_norm = F.normalize(X, p=2, dim=-1)
        f_X_norm = F.normalize(f_X_proj, p=2, dim=-1)

        # Compute weighted bisector
        fusion = self.weight_params[0] * X_norm + self.weight_params[1] * f_X_norm
        fusion = F.normalize(fusion, p=2, dim=-1)

        return fusion

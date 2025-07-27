from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .fusion import (
    ProjectionModule,
    WeightedFusion,
)
from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .loss_func import (
    ContrastiveLoss,
    mse_loss,
    sce_loss,
)
from .quantizer import VQ


def setup_module(
    m_type,
    enc_dec,
    in_dim,
    num_hidden,
    out_dim,
    num_layers,
    dropout,
    activation,
    residual,
    norm,
    nhead,
    nhead_out,
    attn_drop,
    negative_slope=0.2,
    concat_out=True,
    **kwargs,
) -> nn.Module:
    if m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            **kwargs,
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim),
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    # * GraphMAE2's GIN
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
        self,
        args,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        num_dec_layers: int,
        num_remasking: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        mask_rate: float = 0.3,
        remask_rate: float = 0.5,
        remask_method: str = "random",
        mask_method: str = "random",
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        loss_fn: str = "byol",
        drop_edge_rate: float = 0.0,
        alpha_l: float = 2,
        replace_rate: float = 0.0,
        # * temperature for contrastive loss
        tau: float = 1.0,
        # * number of negative samples
        N: float = -1.0,
        # * quantization
        commit_score: float = 0.25,
        # * decoder
        out_dim: int = 128,
        # * quantizer
        quantizer: Optional[VQ] = None,
        # * feature fusion
        fusion: str = "weighted",
    ):
        super(PreModel, self).__init__()
        self.args = args  # * for experimenting the model
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l

        self.num_remasking = num_remasking
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method
        # * list of loss function str
        self._loss_fn = loss_fn if isinstance(loss_fn, list) else [loss_fn]
        self.input_dim = in_dim
        self.output_dim = out_dim

        self._token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        # * decoder's input dimension

        self.quantizer = quantizer
        dec_in_dim = self.quantizer.embed_dim
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,  # * output dimension is the same as the input dimension
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if self._remask_method in ["random", "fixed"]:
            self.dec_mask_token = nn.Parameter(torch.zeros(1, dec_in_dim))

        # * fusion module
        # in_dim: input dimension
        # num_hidden: the hidden dimension of the encoder's output
        print(f"=== Fusion module: {fusion} ===")
        if fusion == "projection":
            self.fusion = ProjectionModule(in_dim, num_hidden)
        elif fusion == "weighted":
            self.fusion = WeightedFusion(in_dim, num_hidden, self.args.fusion_lean)

        self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(alpha_l, tau, N)

        self.print_num_parameters()

    def print_num_parameters(self):
        num_encoder_params = [
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        ]
        num_decoder_params = [
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        ]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(
            f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}"
        )

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        if hasattr(self, "dec_mask_token"):
            nn.init.xavier_normal_(self.dec_mask_token)

    def reset_parameters_for_token_(self, seed=42):
        # Save the current random state
        current_state = torch.get_rng_state()

        if seed is not None:
            # Set the seed to ensure independent randomness
            torch.manual_seed(seed)

        # Initialize enc_mask_token using Xavier initialization
        nn.init.xavier_normal_(self.enc_mask_token)

        if hasattr(self, "dec_mask_token"):
            # Initialize dec_mask_token if it exists
            nn.init.xavier_normal_(self.dec_mask_token)

        # Restore the original random state
        torch.set_rng_state(current_state)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    # * loss function
    def setup_loss_fn(self, alpha_l, tau, N):
        criterion = {}
        for loss in self._loss_fn:
            if "mse" in loss:
                print("=== Use mse_loss ===")
                criterion[loss] = partial(mse_loss)
            elif "sce" in loss:
                print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
                criterion[loss] = partial(sce_loss, alpha=alpha_l)
            elif loss == "cont":
                print("=== Use Contrastive loss ===")
                criterion[loss] = ContrastiveLoss(
                    tau=tau,
                    N=N,
                )
            else:
                raise NotImplementedError
        return criterion

    def forward(
        self, g, x, targets=None, epoch=0, drop_g1=None, drop_g2=None
    ):  # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, targets, epoch, drop_g1, drop_g2)

        return loss

    def mask_attr_prediction(self, g, x, targets, epoch, drop_g1=None, drop_g2=None):
        # # if drop_g1 is not None, it's equivalent as sequentially applying node masking, then drop edge
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, x, self._mask_rate
        )
        # * mask_nodes is the index of masked nodes
        sorted_mask_nodes, _ = torch.sort(mask_nodes)
        sorted_keep_nodes, _ = torch.sort(keep_nodes)

        use_g = drop_g1 if drop_g1 is not None else g
        # * g is type of dgl.heterograph.DGLGraph and kept as a batched graph with proper batch_size

        # * masked graph passes through encoder
        masked_rep = self.encoder(
            use_g,
            use_x,
        )
        masked_rep = self.fusion(use_x, masked_rep)

        # * original graph also passes through encoder
        ori_rep = self.encoder(
            use_g,
            x,
        )
        ori_rep = self.fusion(x, ori_rep)

        # quantization with KL divergence based on raw feature
        commit_loss_ori, ori_rep_q, _ = self.quantizer(ori_rep, raw_feat=x)
        ori_recon = self.decoder(pre_use_g, ori_rep_q)
        loss_rec = sce_loss(ori_recon, x, self._alpha_l)

        loss_cont = 0
        # * quantization
        commit_loss_masked, masked_rep_q, _ = self.quantizer(masked_rep, raw_feat=x)

        # Perform remasking and decoding outside the inner loss loop
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = masked_rep_q.clone()
                remasked_rep, remask_nodes, rekeep_nodes = self.random_remask(
                    use_g, rep, self._remask_rate
                )
                recon = self.decoder(pre_use_g, remasked_rep)
                # rec loss is calculated only for the masked nodes
                for loss in self._loss_fn:
                    if loss in ["mse", "sce", "cont"]:
                        x_init = x[mask_nodes]
                        x_rec = recon[mask_nodes]
                        if loss == "cont":
                            loss_cont += self.criterion[loss](
                                g, recon, x, sorted_mask_nodes
                            )
                        else:
                            loss_rec += self.criterion[loss](x_rec, x_init)
            loss_cont /= self._num_remasking

        elif self._remask_method in ["fixed", "none"]:
            rep = masked_rep_q.clone()
            if self._remask_method == "fixed":
                remasked_rep = self.fixed_remask(g, rep, mask_nodes)
                recon = self.decoder(pre_use_g, remasked_rep)
            else:  # self._remask_method == "none"
                # no remasking
                recon = self.decoder(pre_use_g, rep)
            # rec loss is calculated only for the masked nodes
            for loss in self._loss_fn:
                if loss in ["mse", "sce", "cont"]:
                    x_rec = recon[mask_nodes]
                    x_init = x[mask_nodes]
                    if loss == "cont":
                        loss_cont += self.criterion[loss](
                            g, recon, x, sorted_mask_nodes
                        )
                    else:
                        loss_rec += self.criterion[loss](x_rec, x_init)

        loss = loss_rec + loss_cont + (commit_loss_ori + commit_loss_masked) / 2

        loss_dict = {
            "total_loss": loss,
            "loss_rec": loss_rec,
            "loss_cont": loss_cont,
            "commit_loss_ori": commit_loss_ori,
            "commit_loss_masked": commit_loss_masked,
        }
        return loss_dict

    def embed(self, g, x):
        return self.fusion(x, self.encoder(g, x))

    def embed_quant(self, g, x):
        rep_q, tokens = self.quantizer(self.fusion(x, self.encoder(g, x)), True)
        return rep_q, tokens

    def embed_prompt_quant(self, g, x, prompt):
        x = self.fusion(x, self.encoder(g, x))
        # combine prompt and x with element-wise multiplication
        x = x * prompt
        rep_q, tokens = self.quantizer(x, True)
        return rep_q, tokens

    def prompt_quant(self, x, prompt):
        x = x * prompt
        rep_q, tokens = self.quantizer(x, True)
        return rep_q, tokens

    def get_encoder(self):
        # self.encoder.reset_classifier(out_size)
        return self.encoder

    def get_quantizer(self):
        return self.quantizer

    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict

    # * GraphMAE2's encoding_mask_noise
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # exclude isolated nodes
        # isolated_nodes = torch.where(g.in_degrees() <= 1)[0]
        # mask_nodes = perm[: num_mask_nodes]
        # mask_nodes = torch.index_fill(torch.full((num_nodes,), False, device=device), 0, mask_nodes, True)
        # mask_nodes[isolated_nodes] = False
        # keep_nodes = torch.where(~mask_nodes)[0]
        # mask_nodes = torch.where(mask_nodes)[0]
        # num_mask_nodes = mask_nodes.shape[0]

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    # * GraphMAE's encoding_mask_noise
    def encoding_mask_noise_mae(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[
                perm_mask[: int(self._token_rate * num_mask_nodes)]
            ]
            noise_nodes = mask_nodes[
                perm_mask[-int(self._replace_rate * num_mask_nodes) :]
            ]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes
            ]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[:num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep

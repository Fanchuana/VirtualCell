import logging
from typing import Dict, Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from typing import Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora, load_pretrained_vae
from .vae import *
from timm.loss import AsymmetricLossMultiLabel
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class BioAlignedLoss(nn.Module):
    """
    [TODO]
    Combined Sinkhorn + L_mean + L_cov
    """

    def __init__(self, sinkhorn_weight=1.0, mean_weight=1.0, cov_weight=1.0, blur=0.05, sinkhorn_backend="auto"):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.mean_weight = mean_weight
        self.cov_weight = cov_weight
        self.loss_dict = {}   # 记录损失值，输入日志

        if sinkhorn_weight > 0:
            self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur, backend=sinkhorn_backend)

    @classmethod
    def from_loss_kwargs(cls, kwargs):
        sinkhorn_weight = kwargs.get("sinkhorn_weight", 1)
        mean_weight = kwargs.get("mean_weight", 1)
        cov_weight = kwargs.get("cov_weight", 1)
        sinkhorn_backend = kwargs.get("sinkhorn_backend", "auto")
        blur = kwargs.get("blur", 0.05)

        return cls(
            sinkhorn_weight=sinkhorn_weight, 
            mean_weight=mean_weight, 
            cov_weight=cov_weight, 
            blur=blur, 
            sinkhorn_backend=sinkhorn_backend
        )

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        device = pred.device
        total_loss = torch.zeros(batch_size, device=device)

        # L_sd
        if self.sinkhorn_weight > 0:
            sinkhorn_loss = self.sinkhorn_loss(pred, target) 
            total_loss += self.sinkhorn_weight * sinkhorn_loss
            self.loss_dict["l_sd"] = sinkhorn_loss.nanmean().item()

        # L_mean
        if self.mean_weight > 0:
            # 对一个sentence内的S个细胞做平均 (B, S, D)
            mu_pred = pred.mean(dim=1)
            mu_target = target.mean(dim=1)
            # L1距离
            mean_loss = torch.abs(mu_pred - mu_target).sum(dim=-1)
            total_loss += self.mean_weight * mean_loss
            self.loss_dict["l_mean"] = mean_loss.nanmean().item()
        
        # L_cov
        if self.cov_weight > 0:
            cov_loss = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                # 预测值的矩阵协方差
                cov_pred = torch.cov(pred[b].T)
                # 真实值的矩阵协方差
                cov_target = torch.cov(target[b].T)
                # 矩阵的F范数
                cov_loss[b] = torch.linalg.matrix_norm(cov_pred - cov_target, ord='fro')
            
            total_loss = total_loss + self.cov_weight * cov_loss
            self.loss_dict["l_cov"] = cov_loss.nanmean().item()

        return total_loss

class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val

class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


class VAETransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        vae_kwargs: dict = None,    # [TODO] vae参数显式指明
        loss_kwargs: dict = None,   # [TODO] loss参数显式指明
        extra_dataset_kwargs: dict = None,     # [TODO] DE参数显式指明
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        # [TODO] 主损失
        # Build the distributional loss from geomloss
        main_loss_kwargs = loss_kwargs.get("main_loss_kwargs", None)
        self.use_main_loss = main_loss_kwargs.get("use_main_loss", True)
        if self.use_main_loss:
            blur = main_loss_kwargs.get("blur", 0.05)
            main_loss_name = main_loss_kwargs.get("type", "energy")
            if main_loss_name == "energy":
                self.main_loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
            elif main_loss_name == "mse":
                self.main_loss_fn = nn.MSELoss()
            elif main_loss_name == "se":
                sinkhorn_weight = main_loss_kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
                energy_weight = main_loss_kwargs.get("energy_weight", 1.0)
                self.main_loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
            elif main_loss_name == "sinkhorn":
                self.main_loss_fn = SamplesLoss(loss="sinkhorn", blur=blur, backend="multiscale")
            elif main_loss_name == "ba":
                self.main_loss_fn = BioAlignedLoss.from_loss_kwargs(main_loss_kwargs)
                print("BA LOSS BUILT:", hasattr(self.main_loss_fn, "loss_dict"))
            else:
                raise ValueError(f"Unknown main loss function: {main_loss_name}")
        
        
        # [TODO] Decode4贡献的vae损失: 
        # X_pred的 vae latent 和 X_target的 vae latent之间的损失
        vae_latent_loss_kwargs = loss_kwargs.get("vae_loss_kwargs", None)
        self.use_vae_latent_loss = vae_latent_loss_kwargs.get("use_vae_loss", True)
        self.vae_latent_loss_weight = vae_latent_loss_kwargs.get("weight", 1)
        # decoder 4 分支的损失 默认MMD
        if self.use_vae_latent_loss:
            blur = vae_latent_loss_kwargs.get("blur", 0.05)
            vae_latent_loss_name = vae_latent_loss_kwargs.get("type", "energy")
            if vae_latent_loss_name == "energy":
                self.vae_latent_loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
            elif vae_latent_loss_name == "ba":
                self.vae_latent_loss_fn = BioAlignedLoss.from_loss_kwargs(vae_latent_loss_kwargs)
            else:
                raise ValueError(f"Unknown branch vae latent loss function: {vae_latent_loss_name}")

        # [TODO] vae的设置 后续传入_build_network
        self.vae_kwargs = vae_kwargs
        self.vae_hidden_dim = vae_kwargs.get("hidden_dim", 696)

        # [TODO] 是否计算DE预测损失
        DE_kwargs = loss_kwargs.get("DE_loss_kwargs", None)
        self.predict_DE = DE_kwargs.get("use_DE_loss", True)
        if self.predict_DE:
            DE_loss_name = DE_kwargs.get("type", "BCE")
            if DE_loss_name == "BCE":
                self.DE_loss_fn = nn.BCEWithLogitsLoss()
            elif DE_loss_name == "ASYM":
                self.DE_loss_fn = AsymmetricLossMultiLabel(
                    gamma_pos=0.0,
                    gamma_neg=4.0,
                    clip=0.05,
                    eps=1e-8
                )
            else:
                raise ValueError(f"Unknown branch DE loss function: {DE_loss_name}")
            self.DE_loss_weight = DE_kwargs.get("weight", 1)

        # [TODO] 是否预测gene变化的direction
        direction_kwargs = loss_kwargs.get("direction_loss_kwargs", None)
        self.predict_direction = direction_kwargs.get("use_direction_loss", True)
        if self.predict_direction:
            self.direction_loss_type = direction_kwargs.get("type", "MSE")
            # MSE: 要求精确预测上调/下调幅度, dataset 保留原真实值
            if self.direction_loss_type == "MSE":
                self.direction_loss_fn = nn.MSELoss()
            elif self.direction_loss_type == "smooth_l1":
                self.direction_loss_fn = nn.SmoothL1Loss()
            elif self.direction_loss_type == "BCE":
                self.direction_loss_fn = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unknown branch direction loss function: {self.direction_loss_type}")
            self.direction_loss_weight = direction_kwargs.get("weight", 1)
        
        # [TODO] 是否计算 DE + direction 的 cons 损失
        cons_kwargs = loss_kwargs.get("cons_loss_kwargs", None)
        self.predict_cons = cons_kwargs.get("use_cons_loss", True)
        self.predict_cons = self.predict_cons and self.predict_direction and self.predict_DE
        if self.predict_cons:
            cons_loss_name = cons_kwargs.get("type", "MSE")
            if cons_loss_name == "MSE":
                self.cons_loss_fn = nn.MSELoss()
            else:
                raise ValueError(f"Unknown branch cons loss function: {cons_loss_name}")
            self.cons_loss_weight = cons_kwargs.get("weight", 1)

        
        # [TODO] 如果用到了decoder 2 or 3, 那么加载其所需的新数据   
        self.use_DE_branch = self.predict_DE or self.predict_direction 
        self.DEStatsManager = DEStatsManager(direction_loss_type=self.direction_loss_type, **extra_dataset_kwargs) if self.use_DE_branch else None
 
        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()
        print("is_gene_space:", is_gene_space)  # [TODO]

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")
        if basal_mapping_strategy == "random" and self.use_batch_token:
            disable_reasons.append("basal mapping strategy is random")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: Optional[int] = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._batch_token_cache: Optional[torch.Tensor] = None

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )
        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        # [TODO] 
        self.vae = load_pretrained_vae(
            input_dim=self.input_dim, # 当前的VAE采用的是2000维高变基因X_hvg输入, 要求is_gene_space为True, 且 data.kwargs.embed_key="X_hvg"
            **self.vae_kwargs    
        )
        # 加入VAE encoder
        self.vae_encoder = self.vae.encoder
        
        # Simple linear layer that maintains the input dimension
        # 将vae编码的特征从 vae_hidden_dim 转换到 transf_hidden_dim 
        if self.use_basal_projection:
            self.vae2transf_dim_encoder = build_mlp(
                in_dim=self.vae_hidden_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.vae2transf_dim_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # 简单的扰动编码器
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        # [TODO] 加入我们的VAE decoder
        self.vae_decoder = self.vae.decoder

        # 将transf处理过后的扰动特征从 transf_hidden_dim 转换到 vae_hidden_dim
        if self.use_basal_projection:
            self.transf2vae_dim_decoder = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.vae_hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.transf2vae_dim_decoder = nn.Linear(self.input_dim, self.hidden_dim)

        # [TODO] DE Decoder
        # 将transf特征映射为2000维的gene 0/1 vector. 0: not DE, 1: DE
        if self.predict_DE:
            self.DE_decoder = MultiLabelModelBase(
                input_dim=self.hidden_dim,
                num_classes=self.input_dim,     # 由于我们设置了embed_key="X_hvg", 这里input_dim = output_dim
            )
        
        # [TODO] direction Decoder
        if self.predict_direction:
            self.direction_decoder = MultiLabelModelBase(
                input_dim=self.hidden_dim,
                num_classes=self.input_dim,     # 由于我们设置了embed_key="X_hvg", 这里input_dim = output_dim
            )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.vae_hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def vae_enordecode(self, vae_input: torch.Tensor, vae_layer) -> torch.Tensor:
        """
        调用vae encoder or decoder,
        封装了vae所必需的维度展平+恢复
        """
        # 需要注意的是，VAE其实看不见sentence，它只是对每个细胞都单独做编码
        # 事实上，VAE内部的nn.BatchNorm1d也硬性提示了这一点
        # 因此，我们要在这里先将输入展平，送入VAE之后，再将结果展回
        shape_3d = vae_input.shape
        vae_input_flat = vae_input.reshape(-1, vae_input.size(-1))
        # 调用VAE
        vae_output_flat = vae_layer(vae_input_flat)
        # 展回encoder认识的(B, S, D)
        vae_output = vae_output_flat.reshape(shape_3d[0], shape_3d[1], -1)

        return vae_output

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        # [TODO] 加入VAE 编码HVG 随后用一个简单的MLP 转换为transf的维度
        vae_latent = self.vae_enordecode(expr, self.vae_encoder)

        return self.vae2transf_dim_encoder(vae_latent)
    
    def decode_perted_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we decoder perturbed state input, if needed."""
        # [TODO] 使用VAE 从transf的输出中解码出HVG
        # 类似encoder, 进行展平操作
        vae_latent = self.transf2vae_dim_decoder(expr)
        # [???] expr = self.vae_enordecode(expr, self.vae_decoder)
        expr = self.vae_enordecode(vae_latent, self.vae_decoder)

        return expr, vae_latent

    def forward(self, batch: dict, padded=True) -> ModelForwardOutput:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)

        # [TODO] 
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            # Prepend the batch token to the sequence along the sequence dimension
            # [B, S, H] -> [B, S+1, H], batch token at position 0
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E] (might be one more if we have the batch token)
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)
            
            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract outputs accounting for optional prepended batch token and optional confidence token at the end
        # [TODO] 我们将res_pred重命名为exp_pred
        # 这里不能重建残差, 因为VAE要做的就是重建输入, 很显然我们的输入不是残差, 而是X_ctrl + pert
        if self.confidence_token is not None and self.use_batch_token and self.batch_token is not None:
            # transformer_output: [B, 1 + S + 1, H] -> batch token at 0, cells 1..S, confidence at -1
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            exp_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            # exp_pred currently excludes the confidence token and starts from former index 1
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            # Only confidence token appended at the end
            exp_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            # Only batch token prepended at the beginning
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            exp_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            # Neither special token used
            exp_pred = transformer_output
            self._batch_token_cache = None

        # [TODO] VAE Decoder
        # main branch
        # transf latent -> HVG expr
        out_pred, vae_latent_pred = self.decode_perted_expression(exp_pred)

        # [TODO] DE p-value Decoder
        # decoder2
        # transf latent -> p_vals_logits -> [0(is DE) or 1(not DE)]
        p_vals_logits = self.DE_decoder(exp_pred) if self.predict_DE else None
        
        # [TODO] direction Decoder
        # decoder3
        # transf latent -> direction_logits -> [0(is DE) or 1(not DE)]
        direction_logits = self.direction_decoder(exp_pred) if self.predict_direction else None
        
        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        # logger.info(f"DEBUG: is_gene_space: {is_gene_space}")
        # logger.info(f"DEBUG: self.gene_decoder: {self.gene_decoder}")
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        gene_pred = out_pred.reshape(-1, self.output_dim)

        return ModelForwardOutput(
            gene_pred=gene_pred,
            vae_latent_pred=vae_latent_pred,
            p_vals_logits=p_vals_logits,
            direction_logits=direction_logits,
            confidence_pred=confidence_pred,
        )

    def _cal_and_record_branch_loss(
        self,
        pred,
        gt,
        loss_fn,
        loss_weight,
        branch_name:str = "vae",
        stage:str = "train",
    ):
        """
        分支损失计算 + 日志记录
        """
        if stage == "train":
            branch_loss = loss_fn(pred, gt).nanmean()
        else:
            branch_loss = loss_fn(pred, gt).mean()     
        branch_loss = loss_weight * branch_loss       
        if stage == "train" or stage == "test":
            self.log(
                f"{stage}_{branch_name}_loss", 
                branch_loss,
                prog_bar=True,   # 进度条显示
                on_step=(stage == "train"),
                on_epoch=(stage != "train")
            )
        return branch_loss

    # [TODO] 
    def cal_our_loss(
        self, 
        pred_pack: ModelForwardOutput,
        gt_pack: ModelGroundTruth,    
        stage:str="train", 
    ):
        """
        这里计算我们各个分支的损失: 即除开main_loss, decoder_loss 以外我们自定义的 vae_latent_loss, 等等
        合并写成一个方法, 方便training/val/test_step调用
        Args:
            pred_pack: 打包好的预测值
            gt_pack: 打包好的真实值
        """
        # 初始化 total_loss 为 0
        total_loss = torch.tensor(0.0, device=self.device)

        if self.use_main_loss:
            pred = pred_pack.gene_pred
            target = gt_pack.gene_gt
            # 计算主要损失 X_target and X_pred
            if stage == "train":
                main_loss = self.main_loss_fn(pred, target).nanmean()
            else:
                main_loss = self.main_loss_fn(pred, target).mean()
            # log
            if stage == "train" or stage == "test":
                if hasattr(self.main_loss_fn, "sinkhorn_loss") and hasattr(self.main_loss_fn, "energy_loss"):
                    sinkhorn_component = self.main_loss_fn.sinkhorn_loss(pred, target).nanmean()
                    energy_component = self.main_loss_fn.energy_loss(pred, target).nanmean()
                    self.log(f"{stage}/sinkhorn_loss", sinkhorn_component)
                    self.log(f"{stage}/energy_loss", energy_component)
                # log our ba loss
                if hasattr(self.main_loss_fn, "loss_dict"):
                    ba_loss_dict = self.main_loss_fn.loss_dict
                    self.log_dict(
                        ba_loss_dict, 
                        prog_bar=True,   
                        on_step=(stage == "train"),
                        on_epoch=(stage != "train")
                    )
                else:
                    self.log(
                        f"{stage}_main_loss", 
                        main_loss,
                        prog_bar=True,
                        on_step=(stage == "train"),
                        on_epoch=(stage != "train")
                    )
            total_loss += main_loss
        
        # 计算分支损失
        # decoder4: vae latent对齐损失
        if self.use_vae_latent_loss:
            pred_vae_latent = pred_pack.vae_latent_pred
            target_vae_latent = self.vae_enordecode(target, self.vae_encoder)
            total_loss += self._cal_and_record_branch_loss(
                pred=pred_vae_latent,
                gt=target_vae_latent,
                loss_fn=self.vae_latent_loss_fn,
                loss_weight=self.vae_latent_loss_weight,
                branch_name="vae",
                stage=stage,
            )
        
        # decoder2: DE prediction loss
        if self.predict_DE:
            p_vals_pred = pred_pack.p_vals_logits
            p_vals_gt = gt_pack.p_vals_gt
            total_loss += self._cal_and_record_branch_loss(
                pred=p_vals_pred,
                gt=p_vals_gt,
                loss_fn=self.DE_loss_fn,
                loss_weight=self.DE_loss_weight,
                branch_name="DE",
                stage=stage,
            )

        # decoder3: direction loss
        if self.predict_direction:
            direction_pred = pred_pack.direction_logits
            direction_gt = gt_pack.direction_gt
            total_loss += self._cal_and_record_branch_loss(
                pred=direction_pred,
                gt=direction_gt,
                loss_fn=self.direction_loss_fn,
                loss_weight=self.direction_loss_weight,
                branch_name="direction",
                stage=stage,
            )
        
        # DE + direction cons loss
        if self.predict_cons:
            p_vals_pred = pred_pack.p_vals_logits
            direction_pred = pred_pack.direction_logits

            p_de = torch.sigmoid(p_vals_pred)
            tau, sigma = 0.5, 0.3
            p_from_fc = torch.sigmoid((direction_pred.abs() - tau) / sigma).detach()

            total_loss += self._cal_and_record_branch_loss(
                pred=p_de,
                gt=p_from_fc,
                loss_fn=self.cons_loss_fn,
                loss_weight=self.cons_loss_weight,
                branch_name="cons",
                stage=stage,
            )

        # 总损失 记录以用于best.ckpt
        self.log(
            f"{stage}_loss", 
            total_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=(stage != "train")
        )

        return total_loss

    def _reshape_by_padded(
        self,
        tensor,
        padded: bool = False,
    ):
        """
        简单封装了一下 STATE 的维度操作
        Args:
            padded: reshape成 (B, S, D), 否则展平为 (1, B * S, D)
        """
        if padded:
            return tensor.reshape(-1, self.cell_sentence_len, self.output_dim)
        return tensor.reshape(1, -1, self.output_dim)

    def _get_preds_and_target(
        self,
        batch, 
        pred_pack: ModelForwardOutput, 
        padded: bool = False,
    ):
        """
        简单封装了一下 STATE 对 gene 预测维度的处理
        """
        pred_pack.gene_pred = self._reshape_by_padded(pred_pack.gene_pred, padded)
        target = batch["pert_cell_emb"]
        target = self._reshape_by_padded(target, padded)

        return pred_pack.gene_pred, target

    def _get_p_vals_and_direction(
        self,
        batch, 
        pred_pack: ModelForwardOutput, 
        padded: bool = False,
    ):
        """
        简单封装了一下 DEStatsManager 提取 pert_name 和 cell_type 的操作
        并封装了对 gene 预测维度的处理
        虽然每个 batch 都具有相同的 cell_type 和 pert_name
        不过, batch["pert_name"] 和 batch["cell_type"] 是长为 batch_size * sentence_length 的列表
        也就是说 cell-load 已经帮我们做了 * sentence_length 的复制操作了
        DEStatsManager.get_batch_gt 得到的是 (batch_size * sentence_length, n_genes) 的 tensor
        """
        DE_gt, dir_gt = None, None
        if self.use_DE_branch:
            pred_pack.p_vals_logits = self._reshape_by_padded(pred_pack.p_vals_logits, padded)
            pred_pack.direction_logits = self._reshape_by_padded(pred_pack.direction_logits, padded)

            cell_type_list = batch["cell_type"]        
            pert_name_list = batch["pert_name"]
            DE_gt_raw, dir_gt_raw = self.DEStatsManager.get_batch_gt(cell_type_list, pert_name_list)

            # 以 batch 中已有的张量所在设备为参考
            target_device = batch["pert_cell_emb"].device
            DE_gt = self._reshape_by_padded(DE_gt_raw, padded).to(target_device)
            dir_gt = self._reshape_by_padded(dir_gt_raw, padded).to(target_device)

        return DE_gt, dir_gt

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # [TODO] Get model predictions
        pred_pack = self.forward(batch, padded=padded)
        # 路边
        confidence_pred = pred_pack.confidence_pred
        # gene预测值与真实值
        pred, target = self._get_preds_and_target(batch, pred_pack, padded)
        # DE真实值与direction真实值
        p_vals_gt, direction_gt = self._get_p_vals_and_direction(batch, pred_pack, padded)

        gt_pack = ModelGroundTruth(
            gene_gt=target,
            p_vals_gt=p_vals_gt,
            direction_gt=direction_gt,
        )

        total_loss = self.cal_our_loss(pred_pack, gt_pack, "train")

        # Process decoder if available
        decoder_loss = None

        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]

            B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                # One-hot labels; reshape to [B, S, C]
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                # Integer labels; reshape to [B, S]
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = []
                    for idx in bad_indices:
                        labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]

            # Safety: ensure exactly one target per sequence
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.main_loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())

            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss

            # Add to total loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        # [TODO] Get model predictions
        pred_pack = self.forward(batch)
        # 路边
        confidence_pred = pred_pack.confidence_pred
        # gene预测值与真实值
        # 需要注意的是, 我们的 cal_our_loss 读取的是打包好的 pred_pack 和 gt_pack
        # 因此这里对于 gene_pred 的维度变换，要塞回 pred_pack.gene_pred 去
        pred, target = self._get_preds_and_target(batch, pred_pack, padded=True)
        # DE 真实值与 direction 真实值
        p_vals_gt, direction_gt = self._get_p_vals_and_direction(batch, pred_pack, padded=True)

        gt_pack = ModelGroundTruth(
            gene_gt=target,
            p_vals_gt=p_vals_gt,
            direction_gt=direction_gt,
        )

        loss = self.cal_our_loss(pred_pack, gt_pack, "val")

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.main_loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # [TODO] Get model predictions
        pred_pack = self.forward(batch, padded=False)
        # 路边
        confidence_pred = pred_pack.confidence_pred
        # gene 预测值与真实值
        pred, target = self._get_preds_and_target(batch, pred_pack, padded=False)
        # DE真实值与 direction 真实值
        p_vals_gt, direction_gt = self._get_p_vals_and_direction(batch, pred_pack, padded=False)

        gt_pack = ModelGroundTruth(
            gene_gt=target,
            p_vals_gt=p_vals_gt,
            direction_gt=direction_gt,
        )

        loss = self.cal_our_loss(pred_pack, gt_pack, "test")

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        pred_pack = self.forward(batch, padded=padded)
        # DE真实值与direction真实值
        p_vals_gt, direction_gt = self._get_p_vals_and_direction(batch, pred_pack, padded=padded)

        latent_output = pred_pack.gene_pred
        confidence_pred = pred_pack.confidence_pred

        p_vals_pred = torch.sigmoid(pred_pack.p_vals_logits) if self.predict_DE else None 
        
        p_vals_gt, direction_gt = self._get_p_vals_and_direction(batch, pred_pack, padded=padded) if self.predict_DE and self.predict_direction else None, None

        output_dict = {
            "preds": latent_output,
            "preds_vae_latent": pred_pack.vae_latent_pred,
            "p_vals_pred": p_vals_pred,
            "p_vals_gt": p_vals_gt,
            "direction_pred": pred_pack.direction_logits,
            "direction_gt": direction_gt,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

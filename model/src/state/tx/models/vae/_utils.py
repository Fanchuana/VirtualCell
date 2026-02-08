from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelForwardOutput:
    """
    模型输出的数据类，统一管理所有可能的返回值
    """
    gene_pred: torch.Tensor
    vae_latent_pred: torch.Tensor
    p_vals_logits: Optional[torch.Tensor] = None
    direction_logits: Optional[torch.Tensor] = None
    confidence_pred: Optional[torch.Tensor] = None

@dataclass
class ModelGroundTruth:
    """
    训练时 GroundTruth 的数据类, 统一管理所有可能的返回值
    """
    gene_gt: torch.Tensor
    ctrl: torch.Tensor
    p_vals_gt: Optional[torch.Tensor] = None
    direction_gt: Optional[torch.Tensor] = None

def get_preds_and_target(
        batch, 
        pred_pack: ModelForwardOutput, 
        cell_sentence_len, 
        output_dim,
        padded: bool = False,
    ):
    """
    [已弃用]
    简单封装了一下 STATE 对 gene 预测维度的处理
    """
    pred = pred_pack.gene_pred
    target = batch["pert_cell_emb"]
    if padded:
        pred = pred.reshape(-1, cell_sentence_len, output_dim)
        target = target.reshape(-1, cell_sentence_len, output_dim)
    else:
        pred = pred.reshape(1, -1, output_dim)
        target = target.reshape(1, -1, output_dim)
    return pred, target

def get_p_vals_and_direction(
        batch, 
        statsManager, 
        cell_sentence_len, 
        output_dim,
        padded: bool = False,
        use_DE_branch: bool = False,
    ):
    """
    [已弃用]
    简单封装了一下 DEStatsManager 提取 pert_name 和 cell_type 的操作
    我们知道，每个 batch 都具有相同的 cell_type 和 pert_name
    因此 batch["pert_name"] 和 batch["cell_type"] 就是长为 batch_size 的列表
    DEStatsManager.get_batch_gt 得到的就是 (batch_size, n_genes) 的 tensor
    很显然，这与我们的预测值维度 (batch_size, sentence_length, n_genes) 不匹配
    为了实现逐细胞的对齐，这里需要手动扩展一下维度
    其实还可以考虑通过求均值来压缩预测值维度至 (batch_size, n_genes), 但这样似乎会损失预测精度, 舍弃这个方案
    """
    DE_gt, dir_gt = None, None
    if use_DE_branch:
        cell_type_list = batch["cell_type"]        
        pert_name_list = batch["pert_name"]

        DE_gt_raw, dir_gt_raw = statsManager.get_batch_gt(cell_type_list, pert_name_list)

        if padded:
            pred = DE_gt_raw.reshape(-1, cell_sentence_len, output_dim)
            target = DE_gt_raw.reshape(-1, cell_sentence_len, output_dim)
        else:
            pred = DE_gt_raw.reshape(1, -1, output_dim)
            target = target.reshape(1, -1, output_dim)

        DE_gt = DE_gt_raw.unsqueeze(1).expand(-1, cell_sentence_len, -1)
        dir_gt = dir_gt_raw.unsqueeze(1).expand(-1, cell_sentence_len, -1)

    return DE_gt, dir_gt
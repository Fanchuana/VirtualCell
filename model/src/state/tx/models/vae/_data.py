import torch
import anndata as ad
import numpy as np
import logging
import h5py
from typing import Dict, List

logger = logging.getLogger(__name__)

def byte2str(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

class DEStatsManager:
    """
    加载和管理 Decoder 2/3 所需的 Ground Truth (P-value & LogFC)。
    """
    def __init__(
        self,
        main_h5ad_path: str,
        new_h5ad_path: str,
        direction_loss_type: str = "MSE",
        p_val_threshold: float = 0.05,
        device: str = "cpu",
        **DE_kwargs,
    ):
        """
        Args:
            cell_type_path_map: 字典，key是cell_type名字, value是对应的h5ad文件路径
            p_val_threshold: 定义DE gene的显著性阈值
        """
        self.stats_cache = {} # 结构: {cell_type: {pert_name: {'deg': tensor, 'direction': tensor}}}
        self.device = device
        self.cell_type_list = ["jurkat", "hepg2", "k562", "rpe1"]
        self.hvg_names = []

        # 加载数据
        self._load_data(main_h5ad_path, new_h5ad_path, direction_loss_type, p_val_threshold)


    def _load_data(self, main_h5ad_path, new_h5ad_path, direction_loss_type, p_val_threshold):
        logger.info("Loading Perturbation Stats (LogFC & P-value) into memory...")

        # 训练数据的高变基因名称列表
        # 我们的数据经过筛选，保证2k个基因相同
        main_data = h5py.File(main_h5ad_path, "r")
        hvg_mask = main_data["/var/highly_variable"][:]
        all_gene_names = main_data["var/gene_name_index"][:]
        hvg_names_raw = all_gene_names[hvg_mask]
        self.hvg_names = [byte2str(name) for name in hvg_names_raw]

        adata = ad.read_h5ad(new_h5ad_path)
        for cell_type in self.cell_type_list:
            self.stats_cache[cell_type] = {}
            try:
                # varm内默认是(n_genes, n_perts), 因此需要进行转置变成 (n_perts, n_genes)
                p_vals_df = adata.varm[f"{cell_type}_" + 'DE_pval'].T
                logfcs_df = adata.varm[f"{cell_type}_" + 'DE_log2FC'].T
                
                # 所有 gene_name 的列表
                # 尽管基因相同，但是我们的数据与旧数据的列顺序不同，因此需要重新排序
                try:
                    p_vals_df = p_vals_df[self.hvg_names]
                    logfcs_df = logfcs_df[self.hvg_names]
                except KeyError as e:
                    logger.error(f"Gene mismatch in {cell_type}! H5AD columns do not match hvg_names.")
                    raise e
                
                # 存入字典
                for pert_name, p_row in p_vals_df.iterrows():
                    l_row = logfcs_df.loc[pert_name]

                    p_vals = p_row.values
                    logfcs = l_row.values

                    # Decoder 2 GT: P-value < p_val_threshold 为 1 (显著), 否则 0
                    deg_labels = (p_vals < p_val_threshold).astype(np.float32)

                    # Decoder 3 GT: 
                    # 若采取 BCE 损失, 则要类似Decoder2, 分为两类
                    # LogFC > 0 为 1 (上调), LogFC <= 0 为 0 (下调/不变)
                    if direction_loss_type == "BCE":
                        dir_labels = (logfcs > 0).astype(np.float32)
                    # 若采取 MSE 损失, 则无需处理, 直接让模型预测 direction 的具体值
                    else:
                        dir_labels = logfcs.astype(np.float32)

                    self.stats_cache[cell_type][pert_name] = {
                        'deg': torch.tensor(deg_labels, device=self.device),
                        'direction': torch.tensor(dir_labels, device=self.device),
                    }
                
                # 最后, 对于 ctrl (即 pert_name 为 non-targeting), 
                # 对于没有扰动的场合, 没有基因是差异表达的(DE label = 0), 并且LogFC 按理说也接近 0
                # 所以全置0, 加入字典
                zero_tensor = torch.zeros(len(self.hvg_names), device=self.device)
                self.stats_cache[cell_type]['non-targeting'] = {
                    'deg': zero_tensor,
                    'direction': zero_tensor,
                }
                    
            except Exception as e:
                logger.error(f"Failed to load stats for {cell_type} from {path}: {e}")
                raise e
        
        logger.info("Perturbation Stats loaded.")

    def get_batch_gt(self, cell_types: List[str], pert_names: List[str]):
        """
        根据当前 batch 的 cell_type 和 pert_name 列表，组装 GT Tensor
        我们的每个 batch 都具有相同的 cell_type 和 pert_name
        最终输出维度 (batch_size, n_genes)
        """
        batch_deg = []
        batch_dir = []

        for ct, pert in zip(cell_types, pert_names):
            data = self.stats_cache[ct][pert]
            ct_str = str(ct)
            pert_str = str(pert)

            # 检查是否存在于缓存中
            if ct_str in self.stats_cache and pert_str in self.stats_cache[ct_str]:
                data = self.stats_cache[ct_str][pert_str]
                batch_deg.append(data['deg'])
                batch_dir.append(data['direction'])
            # 如果是没见过的扰动类型，都视为无差异表达 (同non-targeting, 全置0)
            else:
                zero_tensor = torch.zeros(len(self.hvg_names), device=self.device)
                batch_deg.append(zero_tensor)
                batch_dir.append(zero_tensor)

        # Stack 成 (Batch, n_genes)
        return torch.stack(batch_deg), torch.stack(batch_dir)
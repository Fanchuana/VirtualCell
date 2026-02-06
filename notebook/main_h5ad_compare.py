import scanpy as sc

def process_h5(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    print("="*20 + " h5ad_path " + "="*20)
    print(adata)

    print("==== adata.var 预览 ====")
    print(adata.var.head())

    for part_name in ["X_hvg", "X_state", "X_vci"]:
        print("="*10 + f"{part_name}" + "="*10)
        part_m = adata.obsm[f"{part_name}"]
        print(type(part_m))
        print(part_m.shape)
        print(part_m)

process_h5("/work/home/cryoem666/czx/dataset/STATE/arcinstitute-State-Replogle-Filtered-Dec-6-2025/replogle_concat.h5ad")

process_h5("/work/home/cryoem666/xyf/temp/pycharm/state/gene_perturnb_state/data/replogle.h5ad")
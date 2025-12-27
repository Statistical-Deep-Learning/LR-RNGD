# Low-Rank Robust Node-Level Graph Diffusion


RGAE Training Command Example:

```bash
export PYTHON=python3

CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_robust_graph_vae.py \
  --moddir robustgraphvae_cora_meta_0.25_only_feat_lr2.4 \
  --samdir robustgraphvae_cora_meta_0.25_only_feat_lr2.4 \
  --datadir gcl_embeddings/cora/meta_0.25_all_embs.npy \
  --labeldir gcl_embeddings/cora/meta_0.25_all_ps_labels.npy \
  --adjdir graphattk/cora_meta_adj_0.25.npz \
  --norm 1 \
  --lr 2e-4 \
  --coef_recon 1 \
  --coef_map 0 \
  --factor 0 \
  --factor_edgemap 0 \
  --epoch 50000 \
  --freeze 0 \
  --dataset cora \
  --neighbor_map_dim 2708 \
  --batchsize 2708 \
  --feat_emb_dim 512

```


LDM Training Command Example:

```bash
export PYTHON=python3  

CUDA_VISIBLE_DEVICES=$GPU $PYTHON train.py \
  --batchsize 2708 \
  --modch 64 \
  --moddir unet_1d_cora_meta_0.25_64_robustgvae_encode_all_norm_ema \
  --samdir unet_1d_cora_meta_0.25_64_robustgvae_encode_all_norm_ema \
  --epoch 3000 \
  --interval 500 \
  --intervalplot 500 \
  --nettype unet_1d \
  --inch 1 \
  --outch 1 \
  --inputsize 64 \
  --clsnum 7 \
  --datatype gclemb \
  --datadir cora_latents_2708_robustgvae_250_64_encode.npy \
  --labeldir gcl_embeddings/cora/meta_0.25_all_ps_labels.npy \
  --genum 560 \
  --genbatch 280 \
  --norm 1
```


LDM Sampling Command Example:

```bash
export PYTHON=python3

CUDA_VISIBLE_DEVICES=$GPU $PYTHON sample.py \
  --genum 7200 \
  --genbatch 840 \
  --modch 64 \
  --moddir unet_1d_cora_meta_0.25_64_robustgvae_encode_all_norm_ema \
  --samdir unet_1d_cora_meta_0.25_64_robustgvae_encode_all_norm_ema \
  --epoch 3000 \
  --nettype unet_1d \
  --inch 1 \
  --outch 1 \
  --inputsize 64 \
  --clsnum 7 \
  --datadir cora_latents_2708_robustgvae_250_64_encode.npy \
  --labeldir gcl_embeddings/cora/meta_0.25_all_ps_labels.npy \
  --norm 1
```
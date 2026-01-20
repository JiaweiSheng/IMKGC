# IMKGC

Source code for AAAI2026 paper: [_Information-Theoretic Minimal Sufficient Representation for Multi-Domain Knowledge Graph Completion_](https://github.com/JiaweiSheng/IMKGC/blob/main/AAAI_2026_IMKGC.pdf).

Multi-domain knowledge graph completion (MKGC) seeks to predict missing triples in a target KG by leveraging triples from multiple KGs in different domains (e.g., languages or sources).
Existing studies can conceal task-relevant information in representations, impeding further improvements when scaling to numerous KGs.
To this end, we propose IMKGC, an information-theoretic MKGC framework to learn minimal sufficient representations.
Extensive experiments on 14 KGs in three benchmark datasets demonstrate that IMKGC significantly outperforms previous state-of-the-art methods, especially in redundant scenarios.

# Requirements

```
python 3.9
torch==2.3.0+cu118
torch-geometric==2.5.3
torch-scatter==2.1.2+pt23cu118
numpy==1.26.3
pandas==2.2.2
tqdm==4.66.4
```

# Datasets

We evaluate our model on three datasets: **DBP-5L**, **E-PKG**, and **DWY**.

- The original **DBP-5L** dataset is obtained from the [LSMGA-MKGC](https://github.com/RongchuanTang/LSMGA-MKGC) repository.
- The original **E-PKG** dataset is sourced from the [ss-aga-kgc](https://github.com/amzn/ss-aga-kgc) repository.
- The **DWY** dataset originates from [BootEA](https://github.com/nju-websoft/BootEA) and has been reorganized to suit the MKGC task. The adapted version is also released in this repository.

Note that the pre-trained embeddings were not used in our experiments.

# How to run

To train and test the model, pls run as follows:

#### DBP-5L

```
python run_model.py --dataset dbp5l --round 30  --model imkgc --alpha 0.1 --beta 0.0001 --gamma 0.005 --omega 0.05 --vq_loss_w 1. --lr 0.001 --margin 0.5  --batch_size 300 --reason_step 4  --codebook_ratio 0.8 --epoch_each 2 --commit_loss 0.5 --v dbp5l_vanilla
```

#### E-PKG

```
python run_model.py --dataset depkg --round 50  --model imkgc --alpha 0.05 --beta 0.0001 --gamma 0.005 --omega 0.001 --vq_loss_w 1. --lr 0.001 --margin 0.5  --batch_size 300 --reason_step 2  --codebook_ratio 0.6 --epoch_each 2 --commit_loss 0.5 --v depkg_vanilla
```

#### DWY

```
python run_model.py --dataset dwy --round 30  --model imkgc --alpha 0.1 --beta 0.0001 --gamma 0.005 --omega 0.05 --vq_loss_w 1. --lr 0.001 --margin 0.5  --batch_size 300 --reason_step 3  --codebook_ratio 0.8 --epoch_each 3 --commit_loss 0.5 --v dwy_vanilla
```

# Citation

If you find this code useful, pls cite our work:

```
@inproceedings{sheng2026:IMKGC,
  author       = {Jiawei Sheng and Taoyu Su and Weiyi Yang  and Linghui Wang and Yongxiu Xu and  Tingwen Liu},
  title        = {Information-Theoretic Minimal Sufficient Representation for Multi-Domain Knowledge Graph Completion},
  booktitle    = {Proceedings of AAAI},
  year         = {2026}
}
```

We would also like to thank repositories [LSMGA-MKGC](https://github.com/RongchuanTang/LSMGA-MKGC), [ss-aga-kgc](https://github.com/amzn/ss-aga-kgc), [OpenKE-PyTorch](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch) for their contributions to our work.

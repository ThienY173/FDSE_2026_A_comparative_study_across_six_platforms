# When Do Metadata-Derived Graph Features Help Social Media Popularity Prediction?

Code and results for the FDSE 2026 paper *"When Do Metadata-Derived Graph
Features Help Social Media Popularity Prediction? A Comparative Study
Across Six Platforms"* (Thien-Y Nguyen-Thai, Ai-Nu Huynh-Tran, Hung-Nghiep
Tran — University of Information Technology, VNU-HCM).

This repository contains the full pipeline used to compare a metadata-only
XGBoost baseline against four graph-augmented variants — an unweighted
homophily graph, a numeric-similarity $k$-NN graph, a data-driven homophily
graph with mutual-information-learned relation weights, and a variant
additionally supplying Node2Vec embeddings — across six social media
platforms (Facebook, Instagram, Reddit, TikTok, Twitter, YouTube), five
fixed seeds, and a leakage-audited protocol.

## Data

`multi_platform_social_sentiment_evolution.csv` is the raw public dataset
(Kaggle, Gokhale, *Multi-Platform Social Sentiment Evolution*):
150,000 posts across six platforms, 31 raw attributes, no missing values.
Source: https://www.kaggle.com/datasets/sohumgokhale/multi-platform-social-sentiment-evolution

## Repository structure

```
.
├── data_preprocess/     # Feature engineering, leakage-column removal,
│                        # label construction (top-20% engagement per
│                        # platform), one-hot encoding, per-seed stratified
│                        # 80/20 splits
├── Metadata_train/      # Metadata-only (no_graph) XGBoost baseline:
│                        # shared classifier config used by every variant
├── Network_graph/       # Per-platform notebooks ({platform}_graph.ipynb)
│                        # implementing:
│                        #   - homophily edge construction (chain rule,
│                        #     Algorithm 1), shared by unweighted_homophily
│                        #     and data_driven_homophily
│                        #   - mutual-information relation weighting
│                        #     (training-partition only)
│                        #   - similarity k-NN graph construction
│                        #   - centrality features (degree, betweenness,
│                        #     closeness, PageRank, eigenvector, modularity
│                        #     class)
│                        #   - Node2Vec embeddings (D=64)
│                        #   - leakage audit (train/train, train/test,
│                        #     test/test edge counts)
│                        #   - paired per-seed deltas, classification,
│                        #     grouped SHAP, targeted ablation,
│                        #     error-transition analysis
├── results/             # Per-platform result CSVs (see below)
└── multi_platform_social_sentiment_evolution.csv
```

## Results files (per platform, in `results/`)

Each platform produces the following CSVs (column names as actually
written by the pipeline):

| File (pattern) | Contents |
|---|---|
| `{platform}_full_results.csv` | Absolute accuracy / precision / recall / F1 / ROC-AUC / PR-AUC per seed per variant (5 variants × 5 seeds) |
| `{platform}_summary.csv` | Mean and std of the above, aggregated per variant |
| `{platform}_paired_deltas.csv` | Per-seed deltas vs. `no_graph` (`delta_f1_mean/std/n_positive`, same for `pr_auc`, `precision`, `recall`, `roc_auc`) plus the automatic `classification` label (`positive_utility` / `precision_recall_trade_off` / `negligible_effect` / `negative_effect`) |
| `{platform}_ablation_results.csv` | Absolute metrics per seed for `data_driven_homophily+node2vec64_minus_centrality` and `..._minus_node2vec` |
| `{platform}_error_transitions.csv` | `fp_to_tn_corrected_false_positives`, `tp_to_fn_lost_true_positives`, `fn_to_tp_recovered_positives`, `tn_to_fp_new_errors` per seed per variant, computed against the `no_graph` baseline on identical test rows |
| `{platform}_grouped_shap.csv` | Mean absolute SHAP attribution share per group (`metadata` / `centrality` / `node2vec`) for the `data_driven_homophily+node2vec64` variant |
| `{platform}_graph_statistics.csv` | `n_nodes`, `n_edges`, `density`, `avg_degree`, `max_degree`, and the `n_train_train_edges` / `n_train_test_edges` / `n_test_test_edges` leakage-audit decomposition, per seed per graph variant |
| `{platform}_mi_weights.csv` | Per-seed, per-relation mutual-information-derived weight ($w_a$) for all six homophily relations (`same_author`, `same_location`, `same_topic`, `same_language`, `same_media_type`, `same_toxicity_bin`) |

All files share `platform`, `seed`, and `variant` columns for joining.

## Reproducing the paper's tables

- **Table 1** (dataset & graph statistics) ← `{platform}_graph_statistics.csv`, averaged over the 5 seeds.
- **Table 2** (paired deltas + classification) ← `{platform}_paired_deltas.csv` for all six platforms.
- **Table 3** (grouped SHAP, ablation, error transitions) ← `{platform}_grouped_shap.csv`, `{platform}_ablation_results.csv`, `{platform}_error_transitions.csv`, all for the `data_driven_homophily+node2vec64` variant.

## Fixed configuration (identical across platforms, seeds, and variants)

- **Seeds:** `{42, 123, 2024, 7, 99}`
- **Classifier:** XGBoost, `n_estimators=500`, `learning_rate=0.02`, `max_depth=6`, `min_child_weight=1`, `gamma=0.1`, `subsample=0.8`, `colsample_bytree=0.5`, `scale_pos_weight = n_neg/n_pos` (from the current training partition), `objective="binary:logistic"`
- **Label:** top-20% of total engagement, computed once per platform before splitting
- **Split:** stratified 80/20 per platform per seed, reused identically by all 5 variants
- **Homophily relation weights:** min–max normalized to $[0.5, 2.0]$ from training-partition mutual information (see the paper's Section 4.3 for the cardinality-bias finding)
- **Node2Vec:** $D=64$, walk length 20, 20 walks/node, window 5, min count 1, $p=q=1$

## Citation

If you use this code or these results, please cite:

```bibtex
@inproceedings{nguyenthai2026metadata,
  title     = {When Do Metadata-Derived Graph Features Help Social Media
               Popularity Prediction? A Comparative Study Across Six
               Platforms},
  author    = {Nguyen-Thai, Thien-Y and Huynh-Tran, Ai-Nu and Tran, Hung-Nghiep},
  booktitle = {Proceedings of FDSE 2026},
  year      = {2026}
}
```

## License

Add a license (e.g., MIT for code, CC-BY for results) before or shortly
after submission — none is currently specified in this repository.

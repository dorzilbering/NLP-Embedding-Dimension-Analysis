# Descriptive empirical findings

Generated exclusively from validated raw JSON. Relative changes compare each metric to the same model's native score. These are single-seed observations, not significance claims.

## STSB

- Cosine Spearman: highest observed score 0.604179 (Llama3.2-3B, 1536 dimensions). Half-dimension changes range from +6.74% to +34.88%; smallest-tested-dimension changes range from +0.68% to +22.79%.


Best observed dimension within each model (post-hoc):

| model | task | metric | dimension | score | native_score | delta_score | relative_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | STSB | cosine_spearman | 2048 | 0.430739 | 0.319358 | 0.111381 | 34.876502 |
| Gemma3-4B | STSB | cosine_spearman | 1280 | 0.593109 | 0.555657 | 0.037452 | 6.740141 |
| Llama3.2-3B | STSB | cosine_spearman | 1536 | 0.604179 | 0.540604 | 0.063575 | 11.760041 |
| Phi4-mini | STSB | cosine_spearman | 1536 | 0.360622 | 0.332305 | 0.028316 | 8.521121 |
| Mistral-7B | STSB | cosine_spearman | 2048 | 0.499205 | 0.438397 | 0.060808 | 13.870589 |


Smallest tested dimension retaining at least 95% simultaneously across this task's metrics:

| model | task | metric | threshold_pct | smallest_tested_dimension | dimension_fraction |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | STSB | cosine_spearman | 95 | 256 | 0.062500 |
| Gemma3-4B | STSB | cosine_spearman | 95 | 320 | 0.125000 |
| Llama3.2-3B | STSB | cosine_spearman | 95 | 384 | 0.125000 |
| Phi4-mini | STSB | cosine_spearman | 95 | 384 | 0.125000 |
| Mistral-7B | STSB | cosine_spearman | 95 | 512 | 0.125000 |

## Banking77

- Accuracy: highest observed score 0.825974 (Llama3.2-3B, 1536 dimensions). Half-dimension changes range from +4.39% to +68.58%; smallest-tested-dimension changes range from +2.07% to +59.29%.

- Macro-F1: highest observed score 0.823923 (Llama3.2-3B, 1536 dimensions). Half-dimension changes range from +4.94% to +76.27%; smallest-tested-dimension changes range from +2.58% to +65.97%.


Best observed dimension within each model (post-hoc):

| model | task | metric | dimension | score | native_score | delta_score | relative_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | Banking77 | accuracy | 2048 | 0.771753 | 0.457792 | 0.313961 | 68.581560 |
| Qwen3-8B | Banking77 | macro_f1 | 2048 | 0.763507 | 0.433151 | 0.330357 | 76.268296 |
| Gemma3-4B | Banking77 | accuracy | 1280 | 0.821104 | 0.783766 | 0.037338 | 4.763877 |
| Gemma3-4B | Banking77 | macro_f1 | 1280 | 0.819116 | 0.775068 | 0.044049 | 5.683191 |
| Llama3.2-3B | Banking77 | accuracy | 1536 | 0.825974 | 0.791234 | 0.034740 | 4.390644 |
| Llama3.2-3B | Banking77 | macro_f1 | 1536 | 0.823923 | 0.785160 | 0.038763 | 4.936914 |
| Phi4-mini | Banking77 | accuracy | 1536 | 0.729545 | 0.479545 | 0.250000 | 52.132701 |
| Phi4-mini | Banking77 | macro_f1 | 1536 | 0.720760 | 0.452086 | 0.268674 | 59.429781 |
| Mistral-7B | Banking77 | accuracy | 2048 | 0.813636 | 0.749675 | 0.063961 | 8.531832 |
| Mistral-7B | Banking77 | macro_f1 | 2048 | 0.809934 | 0.736221 | 0.073713 | 10.012334 |


Smallest tested dimension retaining at least 95% simultaneously across this task's metrics:

| model | task | metric | threshold_pct | smallest_tested_dimension | dimension_fraction |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | Banking77 | all_task_metrics | 95 | 256 | 0.062500 |
| Gemma3-4B | Banking77 | all_task_metrics | 95 | 320 | 0.125000 |
| Llama3.2-3B | Banking77 | all_task_metrics | 95 | 384 | 0.125000 |
| Phi4-mini | Banking77 | all_task_metrics | 95 | 384 | 0.125000 |
| Mistral-7B | Banking77 | all_task_metrics | 95 | 512 | 0.125000 |

## SciFact

- nDCG@10: highest observed score 0.055978 (Gemma3-4B, 1280 dimensions). Half-dimension changes range from -1.67% to +57.72%; smallest-tested-dimension changes range from -90.01% to -22.49%.

- MRR@10: highest observed score 0.052817 (Gemma3-4B, 1280 dimensions). Half-dimension changes range from -7.45% to +74.39%; smallest-tested-dimension changes range from -89.82% to -31.33%.

- Recall@10: highest observed score 0.088167 (Gemma3-4B, 1280 dimensions). Half-dimension changes range from +4.31% to +45.45%; smallest-tested-dimension changes range from -85.13% to +0.00%.

- Recall@100: highest observed score 0.208944 (Gemma3-4B, 1280 dimensions). Half-dimension changes range from -1.62% to +31.64%; smallest-tested-dimension changes range from -51.49% to -20.68%.


Best observed dimension within each model (post-hoc):

| model | task | metric | dimension | score | native_score | delta_score | relative_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | SciFact | ndcg_at_10 | 4096 | 0.022365 | 0.022365 | 0.000000 | 0.000000 |
| Qwen3-8B | SciFact | mrr_at_10 | 4096 | 0.018948 | 0.018948 | 0.000000 | 0.000000 |
| Qwen3-8B | SciFact | recall_at_10 | 2048 | 0.040000 | 0.036667 | 0.003333 | 9.090909 |
| Qwen3-8B | SciFact | recall_at_100 | 2048 | 0.135000 | 0.126667 | 0.008333 | 6.578947 |
| Gemma3-4B | SciFact | ndcg_at_10 | 1280 | 0.055978 | 0.048163 | 0.007815 | 16.226695 |
| Gemma3-4B | SciFact | mrr_at_10 | 1280 | 0.052817 | 0.047426 | 0.005392 | 11.368327 |
| Gemma3-4B | SciFact | recall_at_10 | 1280 | 0.088167 | 0.070833 | 0.017333 | 24.470588 |
| Gemma3-4B | SciFact | recall_at_100 | 1280 | 0.208944 | 0.193944 | 0.015000 | 7.734174 |
| Llama3.2-3B | SciFact | ndcg_at_10 | 1536 | 0.045787 | 0.045220 | 0.000567 | 1.252755 |
| Llama3.2-3B | SciFact | mrr_at_10 | 1536 | 0.038582 | 0.036712 | 0.001870 | 5.094761 |
| Llama3.2-3B | SciFact | recall_at_10 | 1536 | 0.080667 | 0.077333 | 0.003333 | 4.310345 |
| Llama3.2-3B | SciFact | recall_at_100 | 3072 | 0.164222 | 0.164222 | 0.000000 | 0.000000 |
| Phi4-mini | SciFact | ndcg_at_10 | 1536 | 0.025531 | 0.016188 | 0.009343 | 57.716410 |
| Phi4-mini | SciFact | mrr_at_10 | 1536 | 0.023259 | 0.013337 | 0.009922 | 74.392542 |
| Phi4-mini | SciFact | recall_at_10 | 1536 | 0.036111 | 0.026667 | 0.009444 | 35.416667 |
| Phi4-mini | SciFact | recall_at_100 | 1536 | 0.115111 | 0.087444 | 0.027667 | 31.639136 |
| Mistral-7B | SciFact | ndcg_at_10 | 2048 | 0.016005 | 0.014074 | 0.001931 | 13.722559 |
| Mistral-7B | SciFact | mrr_at_10 | 2048 | 0.012897 | 0.012810 | 0.000087 | 0.681537 |
| Mistral-7B | SciFact | recall_at_10 | 2048 | 0.026667 | 0.018333 | 0.008333 | 45.454545 |
| Mistral-7B | SciFact | recall_at_100 | 2048 | 0.087778 | 0.071111 | 0.016667 | 23.437500 |


Smallest tested dimension retaining at least 95% simultaneously across this task's metrics:

| model | task | metric | threshold_pct | smallest_tested_dimension | dimension_fraction |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | SciFact | all_task_metrics | 95 | 4096 | 1.000000 |
| Gemma3-4B | SciFact | all_task_metrics | 95 | 1280 | 0.500000 |
| Llama3.2-3B | SciFact | all_task_metrics | 95 | 1536 | 0.500000 |
| Phi4-mini | SciFact | all_task_metrics | 95 | 1536 | 0.500000 |
| Mistral-7B | SciFact | all_task_metrics | 95 | 2048 | 0.500000 |

## Arxiv-Clustering

- V-measure: highest observed score 0.270578 (Gemma3-4B, 2560 dimensions). Half-dimension changes range from -18.95% to -7.70%; smallest-tested-dimension changes range from -43.03% to -30.99%.

- NMI: highest observed score 0.270578 (Gemma3-4B, 2560 dimensions). Half-dimension changes range from -18.95% to -7.70%; smallest-tested-dimension changes range from -43.03% to -30.99%.

- Adjusted Rand index: highest observed score 0.101376 (Gemma3-4B, 2560 dimensions). Half-dimension changes range from -33.35% to -15.11%; smallest-tested-dimension changes range from -62.27% to -41.92%.


Best observed dimension within each model (post-hoc):

| model | task | metric | dimension | score | native_score | delta_score | relative_change_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | Arxiv-Clustering | v_measure | 4096 | 0.229998 | 0.229998 | 0.000000 | 0.000000 |
| Qwen3-8B | Arxiv-Clustering | nmi | 4096 | 0.229998 | 0.229998 | 0.000000 | 0.000000 |
| Qwen3-8B | Arxiv-Clustering | adjusted_rand | 4096 | 0.081680 | 0.081680 | 0.000000 | 0.000000 |
| Gemma3-4B | Arxiv-Clustering | v_measure | 2560 | 0.270578 | 0.270578 | 0.000000 | 0.000000 |
| Gemma3-4B | Arxiv-Clustering | nmi | 2560 | 0.270578 | 0.270578 | 0.000000 | 0.000000 |
| Gemma3-4B | Arxiv-Clustering | adjusted_rand | 2560 | 0.101376 | 0.101376 | 0.000000 | 0.000000 |
| Llama3.2-3B | Arxiv-Clustering | v_measure | 3072 | 0.225806 | 0.225806 | 0.000000 | 0.000000 |
| Llama3.2-3B | Arxiv-Clustering | nmi | 3072 | 0.225806 | 0.225806 | 0.000000 | 0.000000 |
| Llama3.2-3B | Arxiv-Clustering | adjusted_rand | 3072 | 0.076104 | 0.076104 | 0.000000 | 0.000000 |
| Phi4-mini | Arxiv-Clustering | v_measure | 3072 | 0.151058 | 0.151058 | 0.000000 | 0.000000 |
| Phi4-mini | Arxiv-Clustering | nmi | 3072 | 0.151058 | 0.151058 | 0.000000 | 0.000000 |
| Phi4-mini | Arxiv-Clustering | adjusted_rand | 3072 | 0.039778 | 0.039778 | 0.000000 | 0.000000 |
| Mistral-7B | Arxiv-Clustering | v_measure | 4096 | 0.251973 | 0.251973 | 0.000000 | 0.000000 |
| Mistral-7B | Arxiv-Clustering | nmi | 4096 | 0.251973 | 0.251973 | 0.000000 | 0.000000 |
| Mistral-7B | Arxiv-Clustering | adjusted_rand | 4096 | 0.095825 | 0.095825 | 0.000000 | 0.000000 |


Smallest tested dimension retaining at least 95% simultaneously across this task's metrics:

| model | task | metric | threshold_pct | smallest_tested_dimension | dimension_fraction |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | Arxiv-Clustering | all_task_metrics | 95 | 4096 | 1.000000 |
| Gemma3-4B | Arxiv-Clustering | all_task_metrics | 95 | 2560 | 1.000000 |
| Llama3.2-3B | Arxiv-Clustering | all_task_metrics | 95 | 3072 | 1.000000 |
| Phi4-mini | Arxiv-Clustering | all_task_metrics | 95 | 3072 | 1.000000 |
| Mistral-7B | Arxiv-Clustering | all_task_metrics | 95 | 4096 | 1.000000 |


## Limits

PCA also changes centering and geometry. Cross-model results reflect checkpoints and loading policies together. V-measure and arithmetic NMI are equivalent definitions here. Low SciFact baselines make relative changes numerically large; always read absolute scores alongside ratios. No pooled task score, statistical significance, uncertainty interval, or prediction at untested dimensions is inferred.

# Evaluation results

Labelled resumes: 12

Active components: spaCy (linguistic analysis) (no), Sentence embeddings (semantic matching) (no), Cross-encoder (re-ranking) (no)

## 1. Role matching accuracy

| Method | Top-1 | Top-3 |
|---|---|---|
| TF-IDF only (baseline) | 100% | 100% |
| Embeddings only | n/a (model not installed) |  |
| Skills only | 100% | 100% |
| Hybrid (skills + TF-IDF) | 100% | 100% |
| Hybrid (skills + embeddings) | n/a (model not installed) |  |

## 2. Skill extraction

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Dictionary only | 0.93 | 0.97 | 0.95 |
| Full NLP pipeline | 0.94 | 0.97 | 0.95 |

Re-run after installing all models to fill in the embedding rows.

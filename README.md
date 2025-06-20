## Semi-supervised Discriminative Analysis Clustering (SDAC)

**Semi-supervised Discriminative Analysis Clustering (SDAC)** is a machine learning technique that combines the strengths of supervised and unsupervised learning for clustering tasks. Unlike traditional clustering methods that rely solely on unlabeled data, SDAC leverages a small amount of labeled data to guide the clustering process, improving the quality and interpretability of the resulting clusters.

### Key Features

- **Semi-supervised Learning:** Utilizes both labeled and unlabeled data to enhance clustering performance.
- **Discriminative Analysis:** Focuses on maximizing the separation between clusters using discriminative criteria, often inspired by techniques like Linear Discriminant Analysis (LDA).
- **Improved Cluster Quality:** Incorporates prior knowledge from labeled samples to produce more meaningful and accurate clusters.

### Typical Workflow

1. **Input Data:** A dataset containing both labeled and unlabeled samples.
2. **Initialization:** Use labeled data to initialize cluster centers or discriminative directions.
3. **Iterative Clustering:** Assign unlabeled samples to clusters based on discriminative features, updating cluster assignments iteratively.
4. **Refinement:** Optionally refine clusters using additional constraints or regularization.

### Applications

- Image and speech recognition
- Bioinformatics and genomics
- Text and document clustering
- Any domain where labeled data is scarce but unlabeled data is abundant

### References

- [Semi-supervised Discriminative Analysis Clustering: Theory and Applications](https://doi.org/10.1109/TPAMI.2017.2677439)
- [A Survey on Semi-supervised Clustering](https://arxiv.org/abs/1902.12134)
- [A Review on SemiSupervised Clustering](https://www.sciencedirect.com/science/article/pii/S0020025523002840?casa_token=o5EoHm6-tE0AAAAA:1B9hIdncoMOJWrPs-ug2M2Z-VT8CKuKNR5W5XZIqx4UdS_gJ7d5r-JnjXynmPwNqkd5_VXZWp9Tz)

---

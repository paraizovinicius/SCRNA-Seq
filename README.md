## SCRNA-Seq Analysis

This repository continues another work, available in [brca-scrna-seq](https://github.com/AILAB-CEFET-RJ/brca-scrna-seq/tree/master).
**Semi-supervised Deep Embedded Clustering (SDEC)**

SDEC is a machine learning approach that combines deep learning with clustering, leveraging both labeled and unlabeled data to improve clustering performance.

### Key Features

- Integrates representation learning and clustering in a unified framework.
- Utilizes a small amount of labeled data to guide the clustering process.
- Learns feature embeddings that are more suitable for clustering tasks.
- Can handle high-dimensional and complex data, such as gene expression profiles.

### Workflow

1. **Preprocessing:** Normalize and preprocess the input data.
2. **Representation Learning:** Use a deep neural network (e.g., autoencoder) to learn low-dimensional embeddings.
3. **Clustering:** Apply clustering in the embedding space.
4. **Semi-supervised Guidance:** Incorporate labeled data to refine cluster assignments and network parameters.
5. **Iterative Optimization:** Alternate between updating cluster assignments and network weights until convergence.
6. **Evaluation**: Evaluate cluster quality with Rand Index and NMI score.


## 📊 Overview of Dataset GEO: GSE75688

[GSE75688](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688) is a Gene Expression Omnibus (GEO) dataset widely used in genomics research.

### 📁 Dataset Files

#### 1. **GSE75688 GEO processed Breast Cancer raw TPM matrix.txt.gz**
- **Columns:**
    - **1-3:** Ensemble ID, gene name, gene type
    - **4-17:** Bulk RNA-seq values for 14 patients (some patients have multiple columns)
    - **18-566:** Gene expression levels for 549 single cells

#### 2. **GSE75688 final sample information.txt.gz**
- **Rows:** 528
- **Columns:**
    1. **Cell Identifier:** Format `BCXX_YY` (e.g., `BC01_02` = cell 02 from patient 01)
    2. **Sample Type:** Single cell or bulk read
    3. **Label:** Tumor or non-tumor
    4. **Body System:** Source of the sample
    5. **Cell Type:** Cell classification

> **Example:**  
> `BC01_02` — Cell 02 from patient 01

### 🧬 Patient ID and Cancer Type Mapping

Each sample in the dataset is labeled with a patient ID (e.g., `BC01_02` — patient ID: 01). The cancer subtype for each patient is inferred as follows:

| **Patient ID(s)** | **Cancer Subtype**                                   |
|:------------------|:-----------------------------------------------------|
| 01, 02            | ER+ (Estrogen Receptor Positive)                     |
| 03                | ER+ and HER2+ (Double Positive)                      |
| 04, 05, 06        | HER2+ (Human Epidermal Growth Factor Receptor 2+)    |
| 07–11             | TNBC (Triple-Negative Breast Cancer)                 |



---

## Instalation

You'll need Python version **3.12.7**.  
To set up the environment, create a new Conda environment with the required Python version:

```bash
conda create -n sdac-env python=3.12.7
conda activate sdac-env
```

Next, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

This will ensure all dependencies needed for the experiment are installed.

## Results

### Comparison with Other Approaches

| Method                | Accuracy | F1     | NMI    | RI     |
|-----------------------|----------|--------|--------|--------|
| **SDEC**              | 0.8311   | 0.8518 | 0.3662 | 0.7187 |
| DEC                   | 0.6990   | 0.7036 | 0.1796 | 0.5784 |
| Spectral Clustering   | 0.7689   | 0.7792 | 0.2995 | 0.6440 |
| KMeans (no PCA)       | 0.5884   | 0.7408 | 0.0340 | 0.5147 |

SDEC outperforms other clustering methods across all evaluation metrics, demonstrating the effectiveness of semi-supervised deep embedded clustering for high-dimensional single-cell RNA-seq data.


## References
- [Semi-supervised deep embedded clustering](https://www.sciencedirect.com/science/article/abs/pii/S0925231218312049?casa_token=ohlOuyvtTu8AAAAA:skBSf2VLFcormWGyNrjlWKkRWiWmnvEn_rPFSORYwJ5eXxGvi-7bQ-_yOwvfev3dMR7k8QGZ43d1)
- [A Review on SemiSupervised Clustering](https://www.sciencedirect.com/science/article/pii/S0020025523002840?casa_token=o5EoHm6-tE0AAAAA:1B9hIdncoMOJWrPs-ug2M2Z-VT8CKuKNR5W5XZIqx4UdS_gJ7d5r-JnjXynmPwNqkd5_VXZWp9Tz)

---

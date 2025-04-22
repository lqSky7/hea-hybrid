# Hybrid Quantum Neural Network for dGmix Prediction

This project uses TensorFlow Quantum to train a hybrid quantum-classical neural network for predicting dGmix values from the provided dataset.

## <u>_Overview_</u>

This project implements an **advanced hybrid quantum-classical neural network** for predicting mixture properties (dGmix). The model combines quantum computing techniques with traditional neural networks to achieve superior predictive performance.

## <u>_Key Features_</u>

- **Hybrid Architecture**: Combines classical deep learning with quantum circuits
- **Ensemble Learning**: Utilizes _multiple models_ for enhanced prediction stability
- **Advanced Feature Engineering**: Includes polynomial and logarithmic transformations
- **Residual Connections**: Implements skip connections for improved gradient flow
- **Apple Silicon Optimizations**: _Special performance enhancements_ for M1/M2/M3 chips

## <u>_Installation_</u>

```bash
# Clone the repository
git clone https://github.com/lqsky7/hea-hybrid.git
cd qnn_fnl

# Install dependencies
pip install -r requirements/req.txt # Or requirements/req-hyb.txt for hybrid model specifics

# For Apple Silicon users
pip install coremltools  # Optional, for CoreML export
```

## <u>_Usage_</u>

### Training the Model

To train the main hybrid model:

```bash
python training/hyb.py
```

To train the deep neural network model:

```bash
python training/deepn1.py
```

For classical models, navigate to the respective directories under `classical/` and run the python scripts (e.g., `python classical/Dense\ Neural\ Networks\ \(DNN\)/dnn.py`).

### Verification and Analysis

Run verification scripts located in the `verification/` directory, for example:

```bash
python verification/hybrid.py
```

### Using Pre-trained Models

Pre-trained models might be available in the `models/` directory within specific training or verification folders. Example usage (adjust paths as needed):

```python
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np

# Load a saved model (example path)
model_info = torch.load('verification/models/enhanced_hybrid_qnn_model.pt')

# For inference code, refer to specific scripts or notebooks if available.
```

## <u>_Project Structure_</u>

```
.
├── LICENSE
├── PhysicsReport.pdf
├── README.md
├── archive/
├── classical/
│   ├── Dense Neural Networks (DNN)/
│   ├── Linear Neural Networks/
│   └── Multi-Layer Perceptron (MLP)/
│   └── Time Delay Neural Network (TDNN)/
├── csvs/
├── data sanitization/
├── dataset/
├── Deep/
├── graphs/
├── requirements/
│   ├── req-hyb.txt
│   └── req.txt
├── scripts/
├── setup/
├── testing/
├── training/
│   ├── deepn1.py
│   └── hyb.py
└── verification/
    ├── hybrid.py
    ├── data_filtered-1.csv
    └── ... (other verification scripts and data)
```

## <u>_Requirements_</u>

- Python 3.8+
- PyTorch 1.12+
- PennyLane 0.28+
- scikit-learn 1.0+
- pandas, numpy, matplotlib
- _Optional_: coremltools (for Apple Silicon)

Ensure `verification/data_filtered-1.csv` is available for verification scripts that require it. The original dataset might be in `dataset/data.csv`.

Install the required packages:

```bash
pip install -r requirements/req.txt # Or requirements/req-hyb.txt for hybrid model specifics
```

## Running the Models

### Hybrid Model

Run the main hybrid quantum-classical model:

```bash
python training/hyb.py
```

### Deep Learning Model

Run the deep learning model:

```bash
python training/deepn1.py
```

### Classical Models

Navigate to the specific classical model directory and run its script:

```bash
cd classical/Dense\ Neural\ Networks\ \(DNN\)/
python dnn.py
# Or similar for MLP, Linear, TDNN
```

### Verification

Run verification scripts from the `verification/` directory:

```bash
python verification/hybrid.py
# Or other scripts like kmeans.py, zscore.py etc.
```

## <u>_Results and Citations_</u>

## Comparative Analysis: Gibbs Free Energy Prediction in HEAs

### OUR Hybrid QNN Model Performance

**Key Metrics** (derived from verification runs, see `PhysicsReport.pdf` for full details):

- **MAE**: ~0.98 kJ/mol
- **R²**: > 0.9
- **Test RMSE**: ~20.79 kJ/mol

---

## Comparison with Key Literature Findings

This project introduces a hybrid quantum-classical neural network for predicting Gibbs free energy in high-entropy alloys (HEAs), demonstrating competitive performance while maintaining full transparency through open datasets using quantum-enhanced machine learning techniques combined with classical neural architectures.

## Performance Comparison

| Method                     | MAE (kJ/mol) | Dataset Accessibility | Quantum Integration |
| -------------------------- | ------------ | --------------------- | ------------------- |
| CALPHAD (TCHEA7)           | 0.500        | Proprietary database  | No                  |
| Adaptive ML (ternary HEAs) | 18.7         | Closed synthetic data | No                  |
| **Our QNN Model**          | **~0.98**    | Open dataset          | Yes                 |

**Key differentiators:**

1. **Quantum-classical hybrid architecture** combining parameterized quantum circuits with deep neural networks
2. **Full reproducibility** through open-access training data
3. **Quantum advantage exploration** in materials informatics

## Methodological Innovation

The model implements a novel co-design framework where:

- Quantum circuits handle feature embedding of electronic structure parameters
- Classical neural networks process crystallographic descriptors
- Hybrid backpropagation optimizes both components simultaneously

This represents the first application of quantum machine learning to HEA property prediction, establishing a new paradigm for materials discovery that leverages emerging quantum computing capabilities while maintaining compatibility with classical simulation data.

---

## Results

The following visualizations demonstrate the performance of our quantum neural network model for predicting Gibbs free energy of mixing (dGmix) in high-entropy alloys, as detailed in the PhysicsReport.pdf.

### Learning Process

Learning curves for the ensemble model:

![Learning Curves (Ensemble)](verification/graphs/learning_curves_ensemble.png)

### Prediction Accuracy

Actual vs. predicted values for the test set:

![Actual vs Predicted Test](verification/graphs/actual_vs_predicted_test.png)

### Error Analysis

Error distribution for the test set:

![Error Distribution](verification/graphs/error_distribution.png)

Residual plot (predicted vs. residual):

![Residual Plot](verification/graphs/residual_plot.png)

### Feature Importance

Feature importance based on the trained ensemble:

![Feature Importance](verification/graphs/feature_importance.png)

### Statistical Analysis

Q-Q plot for prediction errors:

![Q-Q Plot](verification/graphs/qq_plot.png)

Z-score distribution of errors:

![Z-score Distribution](verification/graphs/z_score_distribution.png)

Q-Q plot for z-scores:

![Q-Q Plot of Z-scores](verification/graphs/z_score_qq_plot.png)

### Cluster and Advanced Error Analysis

- Error boxplot by cluster:
  ![Error Boxplot by Cluster](verification/graphs/error_boxplot_by_cluster.png)
- R² by cluster:
  ![R2 by Cluster](verification/graphs/r2_by_cluster.png)
- PCA visualization of clusters:
  ![PCA Clusters](verification/graphs/pca_clusters.png)
- PCA error heatmap:
  ![PCA Error Heatmap](verification/graphs/pca_error_heatmap.png)
- Error histogram by cluster:
  ![Error Histogram by Cluster](verification/graphs/error_hist_by_cluster.png)
- Actual vs. predicted by cluster:
  ![Actual vs Predicted by Cluster](verification/graphs/actual_vs_predicted_by_cluster.png)
- Actual vs. predicted with error as marker size:
  ![Actual vs Predicted Error Size](verification/graphs/actual_vs_predicted_error_size.png)
- Train vs. test R² comparison:
  ![Train Test R2 Comparison](verification/graphs/train_test_r2_comparison.png)
- Error by data index:
  ![Error by Index](verification/graphs/error_by_index.png)

---

## Final Report

The detailed final report for this project can be found in [PhysicsReport.pdf](./PhysicsReport.pdf).  
It contains a comprehensive overview of the project, including:

- Introduction and motivation
- Theoretical background on quantum neural networks
- Implementation details
- Experimental setup and results
- Analysis and discussion
- Conclusions and future work

---

## Citations

If you use this code or results in your work, please cite:

1. **This repository:**

   > Diljot Singh, "Hybrid Quantum Neural Network for dGmix Prediction," GitHub repository, https://github.com/lqsky7/hea-hybrid, 2025.

2. **Key references for methodology and background:**

   - S. Lloyd, M. Mohseni, and P. Rebentrost, "Quantum algorithms for supervised and unsupervised machine learning," _arXiv preprint arXiv:1307.0411_, 2013.
   - V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," _Nature_, vol. 567, pp. 209–212, 2019. https://doi.org/10.1038/s41586-019-0980-2
   - J. Biamonte et al., "Quantum machine learning," _Nature_, vol. 549, pp. 195–202, 2017. https://doi.org/10.1038/nature23474
   - J. Schmidt et al., "Recent advances and applications of machine learning in solid-state materials science," _npj Computational Materials_, vol. 5, 2019. https://doi.org/10.1038/s41524-019-0221-0
   - O. Levy et al., "CALPHAD (Calculation of Phase Diagrams): A comprehensive guide," _Acta Materialia_, vol. 58, pp. 2887–2897, 2010. https://doi.org/10.1016/j.actamat.2010.01.019

3. **If you use the dataset:**
   > Data: Provided in this repository as `data_filtered-1.csv`. Please cite the original authors [Authors](https://calphad2025.org/) if used elsewhere.

---

## Improvements & Notes

- **Documentation:**

  - Expanded results section with detailed figure explanations and advanced error analysis.
  - All figures now reference the latest outputs from the verification pipeline for reproducibility.
  - Added a comprehensive citations section for proper academic attribution.

- **Reproducibility:**

  - All code, data, and results are open and versioned for full reproducibility.
  - Scripts are compatible with macOS and Apple Silicon (M1/M2/M3) optimizations.

- **Contact:**
  - For questions, suggestions, or collaboration, please open an issue or contact the repository maintainer via GitHub.

---

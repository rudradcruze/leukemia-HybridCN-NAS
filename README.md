# Leukemia Classification Using HybridCN-NAS

A deep learning-based system for automated classification of acute lymphoblastic leukemia (ALL) subtypes from peripheral blood smear images.

🌐 **Website**: [https://leukemia.francisrudra.com/](https://leukemia.francisrudra.com/)

## Overview

This project implements a novel hybrid feature extraction model (HybridCN-NAS) combining ConvNeXt and NASNet Large architectures with Multi-Head Self-Attention classifier for accurate leukemia subtype classification.

## Project Structure

The project includes various experimental notebooks:

-   **Feature_Extraction.ipynb** - Feature extraction from leukemia images
-   **Feature_Selection.ipynb** - Feature selection methods
-   **Segmentation.ipynb** - Image segmentation preprocessing
-   **Model_Comparison.ipynb** - Comparison of different models
-   **Ablation_Study.ipynb** - Ablation study experiments
-   **Optimizer_Comparison.ipynb** - Optimizer performance analysis
-   **Learning_Rate.ipynb** - Learning rate experiments
-   **Batch_Size.ipynb** - Batch size experiments
-   **Epoch.ipynb** - Epoch analysis
-   **Normalization.ipynb** - Normalization techniques
-   **Feature_Comparison.ipynb** - Feature comparison analysis
-   **UMAP.ipynb** - UMAP visualization

## Dataset

-   **Source**: Taleqani Hospital ALL dataset
-   **Total Images**: 3,256 peripheral blood smear images
-   **Classes**: 4 (Benign Hematogones, Early pre-B ALL, Pre-B ALL, Pro-B ALL)
-   **Image Size**: 224 × 224 pixels (RGB)

## Key Features

-   Attention U-Net based cell segmentation
-   Hybrid feature extraction (ConvNeXt + NASNet Large)
-   mRMR feature selection (2000 features)
-   Multi-Head Self-Attention classifier
-   Web-based deployment using FastAPI

## Performance

-   **Accuracy**: 99.92%
-   **AUC**: 99.64%
-   **F1-Score**: 96.14%
-   **MCC**: 95.06%

## Technology Stack

-   Python
-   PyTorch
-   FastAPI
-   OpenCV, NumPy, Pandas

## Note

**A comprehensive README with detailed installation instructions, usage guidelines, model architecture details, and reproducibility information will be published after the acceptance of the associated research paper currently under review at iScience journal.**

For questions or early access requests, please contact the authors through the repository.

## Citation

```
[Citation will be added upon paper acceptance]
```

## License

[License information will be added upon paper publication]

_This project is part of ongoing research in medical image analysis and AI-assisted diagnosis._

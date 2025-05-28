# Denoising Autoencoder (DAE), Contractive Autoencoder (CAE), and Variational Autoencoder (VAE)

This repository explores **unsupervised representation learning** using three types of autoencoders:

- Denoising Autoencoder (DAE)
- Contractive Autoencoder (CAE)
- Variational Autoencoder (VAE)

Each model is implemented, tuned, and evaluated using rigorous methods such as **K-Fold cross-validation**. Our focus is on learning **robust representations**, **reconstructing input data**, and understanding how different autoencoders behave in terms of **feature extraction, generative ability, and latent space quality**.

---

## 📌 Project Structure

### 1. Denoising Autoencoder (DAE)
We build a DAE that learns the data manifold by reconstructing clean inputs from noisy versions. This encourages generalization and robustness.

- **Input Noise**: Gaussian noise added before encoding
- **Loss**: MSE between clean input and reconstruction
- **Key Hyperparameters**:
  - Learning rate (lr)
  - Batch size (bs)
  - Number of epochs
  - Noise standard deviation (std)
  - Network depth and width
- **Hyperparameter Tuning**:
  - Performed in **stages** using **K-Fold CV**
  - Grouped tuning for performance-efficiency tradeoffs
- **Findings**:
  - Best tradeoff: `bs=64`, `lr=2e-2`
  - Latent dim of 32 achieves good balance
  - Input noise improves generalization but too much causes blur
- **Limitations**:
  - Poor interpolation and generative ability
  - Latent space is discontinuous

### 2. Contractive Autoencoder (CAE)
The CAE adds a **contractive penalty** to the DAE structure, encouraging robustness to input perturbations and improved feature extraction.

- **Loss**: MSE + Frobenius norm of latent Jacobian
- **CAE ≈ DAE + Contractive Loss**
- **Use Case**: Feature extraction for classification
- **Classifiers Used**:
  - SVM (Accuracy: **98.2%**)
  - KNN (Accuracy: **97.5%**)
- **Key Hyperparameters**:
  - Contractive penalty (`λ`)
  - Latent space dimension
  - Encoder/decoder size
- **Findings**:
  - Latent dim ≥ 48 improves linear separability
  - Deeper networks hurt separability
  - Noise + contraction yields robust features

### 3. Variational Autoencoder (VAE)
The VAE is a **generative model** with a **probabilistic latent space**, enabling interpolation and sample generation.

- **Loss**: Reconstruction loss + KL divergence
- **Latent Space**: Continuous and smooth (unlike DAE or CAE)
- **Hyperparameter Tuning**:
  - Latent dimension most critical
  - Beyond dim=60, marginal improvements
- **Findings**:
  - Reconstructions are **blurry but coherent**
  - Sampling and interpolation work as expected
  - KL term enforces structure but limits sharpness

---

## 🧪 Dataset

- All experiments are conducted on the **MNIST** dataset.
- Inputs are flattened 28×28 grayscale images of handwritten digits.

---

## 🛠️ Implementation Notes

- **Framework**: PyTorch
- **Validation**: K-Fold (typically K=5)
- **Training Aids**: Early stopping, cross-validation loss analysis
- **Evaluation**:
  - Reconstruction quality
  - Latent space structure and interpolation
  - Classification using latent features (CAE)

---

## 📊 Results Summary

| Model | Reconstruction | Latent Quality | Generative Ability | Classification Accuracy |
|-------|----------------|----------------|---------------------|--------------------------|
| DAE   | Sharp, robust  | Discontinuous  | Poor                | N/A                      |
| CAE   | Sharp          | Robust, separable | Poor             | **SVM: 98.2%**, **KNN: 97.5%** |
| VAE   | Blurry         | Smooth, continuous | Good             | N/A                      |

---

## 📈 Key Takeaways

- **DAE** teaches the model to generalize input variations.
- **CAE** excels in extracting robust features suitable for classification.
- **VAE** balances reconstruction and generation but sacrifices some sharpness for structure.

---

## 🚀 Future Work

- Combine CAE and VAE to explore both robust features and generative capabilities.
- Apply models to more complex datasets (e.g., CIFAR-10, FashionMNIST).
- Investigate unsupervised clustering performance in latent space.

---

## 📂 Repository Contents
This repository includes all necessary components to reproduce the experiments and results described:

### 🔸 [`AEs.ipynb`](./AEs.ipynb)
- A comprehensive Jupyter notebook that:
  - Implements Denoising Autoencoder (DAE), Contractive Autoencoder (CAE), and Variational Autoencoder (VAE)
  - Performs K-Fold cross-validation for hyperparameter tuning
  - Visualizes reconstructions, latent space interpolations, and sampling results
  - Evaluates classification performance in CAE latent space using SVM and KNN

> 📌 Use this notebook to train, tune, and test all autoencoder models interactively.

---

### 🔸 [`vae_weights.pt`](./vae_weights.pt)
- Pretrained weights of the final Variational Autoencoder (VAE) model.
- Saved as a PyTorch `state_dict` after training on the full MNIST training set.

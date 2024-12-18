# Fine-Tuning Llama 3.2 Vision and MoonDream2
---

## 📂 Included Files

- **`FineTuning_Llama_CatsVsDogs.ipynb`**: Demonstrates fine-tuning using the Llama 3.2 Vision model.
- **`FineTuning_MoonDream2_CatsVsDogs.ipynb`**: Shows the same process with MoonDream2.

---

## 🚀 How It Works

### 🐱 Fine-Tuning on `cats_vs_dogs` Dataset
The `cats_vs_dogs` dataset, available on Hugging Face, is used for demonstration. The dataset is divided into two categories (`cats` and `dogs`) and is preprocessed for training.

### 📄 Notebook Highlights
1. **Dataset Preparation**:
   - Download and preprocess the dataset.
   - Images are resized to 224x224 pixels and converted into tensors.

2. **Model Setup**:
   - Load pre-trained `Llama 3.2 Vision` or `MoonDream2` models.
   - Tokenizers are used to handle labels and textual representations.

3. **Fine-Tuning**:
   - Configure hyperparameters (learning rate, batch size, and epochs).
   - Train the model on a subset of the dataset.

4. **Evaluation**:
   - Evaluate model performance on a validation set.
   - Visualize predictions to validate accuracy.

### ⚠️ Prerequisite: Model Installation
Both `Llama 3.2 Vision` and `MoonDream2` must be installed and accessible locally before running the notebooks. These models can be downloaded from their respective repositories or Hugging Face.

---

## 💻 System Requirements

### 🖥️ Minimum Hardware
- **GPU**: NVIDIA RTX 3090 or higher with at least **24 GB VRAM**.
- **RAM**: 16 GB or more.
- **Storage**: 10 GB free for dataset and model storage.

### 🏎️ Recommended Hardware for Fast Performance
- **GPU**: NVIDIA A100 or V100.
- **RAM**: 32 GB or more.
- **Storage**: NVMe SSD for faster I/O.

---

## 🔧 Software Requirements

### 🛠️ Dependencies
Both notebooks rely on the following libraries:

- Python 3.8 or newer
- `transformers` (Hugging Face)
- `torch` and `torchvision` (PyTorch)
- `unsloth` (for fine-tuning convenience)
- `datasets` (to load and preprocess data)
- `matplotlib` (for visualization)

Install all dependencies using:
```bash
pip install transformers torch torchvision unsloth datasets matplotlib
```

---

## 📝 Notes

- Fine-tuning large models requires significant computational resources. Make sure your system meets the recommended GPU requirements for reduced training times.
- If GPU memory is insufficient, reduce batch size or switch to mixed precision training (not included in these notebooks).
- Ensure the models are pre-installed before running the notebooks.

---





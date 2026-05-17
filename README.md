# Advanced NLP Exercise 1 – Fine Tuning BERT for Paraphrase Detection

This project fine-tunes the pretrained `bert-base-uncased` model for paraphrase detection on the MRPC dataset from the GLUE benchmark using Hugging Face Transformers and PyTorch.

The project supports:
- Training
- Evaluation
- Prediction
- Hyperparameter experimentation
- Weights & Biases logging

---

# Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset

The project uses the MRPC dataset from the GLUE benchmark.

Task:
- Paraphrase Detection

Model:
- `bert-base-uncased`

The dataset is automatically downloaded using the Hugging Face `datasets` library.

---

# General Command Format

```bash
python ex1.py \
--max_train_samples <number_of_train_samples> \
--max_eval_samples <number_of_validation_samples> \
--max_predict_samples <number_of_prediction_samples> \
--lr <learning_rate> \
--num_train_epochs <number_of_epochs> \
--batch_size <batch_size> \
--do_train \
--do_predict \
--model_path <path_to_model>
```

---

# Running the Project

## Train a Model

```bash
python ex1.py \
--do_train \
--max_train_samples -1 \
--max_eval_samples -1 \
--num_train_epochs 2 \
--lr 2e-5 \
--batch_size 16 \
--output_dir ./models/ep2_lr2e-5_bs16
```

This command:
- fine-tunes the model on the training set,
- evaluates the model on the validation set,
- logs train loss to Weights & Biases,
- appends validation accuracy to `res.txt`,
- saves the trained model.

---

## Generate Predictions

```bash
python ex1.py \
--do_predict \
--max_predict_samples -1 \
--batch_size 16 \
--model_path ./models/ep2_lr2e-5_bs16
```

This command:
- loads a trained model,
- generates predictions on the MRPC test set,
- creates `predictions.txt`.

---

## Smoke Test Example

```bash
python ex1.py \
--do_train \
--do_predict \
--max_train_samples 32 \
--max_eval_samples 32 \
--max_predict_samples 32 \
--num_train_epochs 1 \
--lr 1e-4 \
--batch_size 8 \
--output_dir ./models/smoke
```

This small run is useful for quickly verifying that the entire pipeline works correctly.

---

# Command Line Arguments

| Argument | Description |
|---|---|
| `--max_train_samples` | Number of training samples to use, or `-1` for all samples |
| `--max_eval_samples` | Number of validation samples to use, or `-1` for all samples |
| `--max_predict_samples` | Number of prediction samples to use, or `-1` for all samples |
| `--num_train_epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--batch_size` | Batch size |
| `--do_train` | Run training |
| `--do_predict` | Run prediction |
| `--model_path` | Path to a trained model for prediction |
| `--output_dir` | Directory used for saving trained models |

---

# Output Files

## res.txt

Contains validation accuracies for all tested hyperparameter configurations.

Example:

```text
epoch_num: 2, lr: 2e-05, batch_size: 16, eval_acc: 0.8407
```

---

## predictions.txt

Contains predictions for the MRPC test set in the required format:

```text
sentence1###sentence2###predicted_label
```

---

## train_loss.png

Contains the train loss plots exported from Weights & Biases.

---

# Weights & Biases

The project uses Weights & Biases (W&B) for experiment tracking and train loss visualization.

When running the project for the first time, you may be asked to log in using a W&B API key.

---

# Hardware Support

The project supports:
- CUDA GPUs
- Apple Silicon MPS acceleration
- CPU fallback



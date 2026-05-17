import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

try:
    import wandb
except ImportError:
    wandb = None


MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
TASK_NAME = "mrpc"
PREDICTIONS_FILE = "predictions.txt"
RESULTS_FILE = "res.txt"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune BERT on GLUE MRPC paraphrase detection.")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./model")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set a random seed for reproducibility, to help have more stable training results by fixing
    the generating of random numbers
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Select the device to run the model on (CPU of GPU) for better performance
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_samples_from_dataset(dataset, samples_number: int):
    """
    Select a specific number of samples from the dataset,
    if samples_number = -1 we choose the full dataset,
    return a subset of size sample_number of the dataset
    """
    if samples_number is not None and samples_number != -1:
        return dataset.select(range(min(samples_number, len(dataset))))
    return dataset


def prepare_datasets(tokenizer, args: argparse.Namespace):
    """
    Loading and tokenizing the MRPC dataset,
    and returns the splits of the dataset
    """
    raw = load_dataset(DATASET_NAME, TASK_NAME)

    train_dataset = select_samples_from_dataset(raw["train"], args.max_train_samples)
    eval_dataset = select_samples_from_dataset(raw["validation"], args.max_eval_samples)
    predict_dataset = select_samples_from_dataset(raw["test"], args.max_predict_samples)

    def tokenize(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    predict_dataset = predict_dataset.map(tokenize, batched=True)

    columns = ["input_ids", "token_type_ids", "attention_mask", "label"]
    predict_columns = ["input_ids", "token_type_ids", "attention_mask"]

    train_dataset.set_format(type="torch", columns=[c for c in columns if c in train_dataset.column_names])
    eval_dataset.set_format(type="torch", columns=[c for c in columns if c in eval_dataset.column_names])
    predict_dataset.set_format(type="torch", columns=[c for c in predict_columns if c in predict_dataset.column_names])

    return raw, train_dataset, eval_dataset, predict_dataset


def make_dataloader(dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Create a Pytorch DataLoader to group the examples into batches and apply dynamic paddingß
    """
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Move a batch of tensors to the selected device
    """
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluating the model on the validation set
    """
    model.eval()
    preds: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            y = batch.pop("labels") if "labels" in batch else batch.pop("label")
            outputs = model(**batch)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            preds.extend(batch_preds.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    return float(accuracy_score(labels, preds))


def append_result(args: argparse.Namespace, eval_acc: float) -> None:
    """
    Save the validation results for each run in the res.txt
    """
    line = (
        f"epoch_num: {args.num_train_epochs}, "
        f"lr: {args.lr}, "
        f"batch_size: {args.batch_size}, "
        f"eval_acc: {eval_acc:.4f}\n"
    )
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def train(args: argparse.Namespace, tokenizer, train_dataset, eval_dataset, device: torch.device) -> Tuple[str, float]:
    """
    Fine-tune the model of the MRPC training set
    """
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_loader = make_dataloader(train_dataset, tokenizer, args.batch_size, shuffle=True)
    eval_loader = make_dataloader(eval_dataset, tokenizer, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    use_wandb = wandb is not None and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
    if use_wandb:
        wandb.init(
            project="anlp-ex1-mrpc",
            name=f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}",
            config={
                "model": MODEL_NAME,
                "task": TASK_NAME,
                "num_train_epochs": args.num_train_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
            },
        )

    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{args.num_train_epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            if "label" in batch and "labels" not in batch:
                batch["labels"] = batch.pop("label")

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            loss_value = float(loss.detach().cpu().item())
            progress.set_postfix({"loss": loss_value})
            if use_wandb:
                wandb.log({"train/loss": loss_value, "epoch": epoch + 1}, step=global_step)

    eval_acc = evaluate(model, eval_loader, device)
    append_result(args, eval_acc)

    if use_wandb:
        wandb.log({"eval/accuracy": eval_acc})
        wandb.finish()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Validation accuracy: {eval_acc:.4f}")
    print(f"Saved model to: {args.output_dir}")
    return args.output_dir, eval_acc


def predict(args: argparse.Namespace, tokenizer, raw_dataset, predict_dataset, device: torch.device) -> None:
    """
    Generate the prediction for MRPC test set, we use the trained model,
    and write the predictions in the prediction.txt
    """
    if args.model_path is None:
        raise ValueError("--model_path is required when using --do_predict unless you also train first.")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    predict_loader = make_dataloader(predict_dataset, tokenizer, args.batch_size, shuffle=False)
    predictions: List[int] = []

    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Predicting"):
            batch = move_batch_to_device(batch, device)
            batch.pop("label", None)
            batch.pop("labels", None)
            outputs = model(**batch)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_preds.cpu().numpy().tolist())

    original_test = select_samples_from_dataset(raw_dataset["test"], args.max_predict_samples)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        for example, pred in zip(original_test, predictions):
            sentence1 = example["sentence1"].replace("\n", " ").strip()
            sentence2 = example["sentence2"].replace("\n", " ").strip()
            f.write(f"{sentence1}###{sentence2}###{pred}\n")

    print(f"Wrote predictions to {PREDICTIONS_FILE}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer_source = args.model_path if (args.do_predict and args.model_path is not None) else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    raw, train_dataset, eval_dataset, predict_dataset = prepare_datasets(tokenizer, args)

    if args.do_train:
        model_path, _ = train(args, tokenizer, train_dataset, eval_dataset, device)
        if args.do_predict and args.model_path is None:
            args.model_path = model_path

    if args.do_predict:
        predict(args, tokenizer, raw, predict_dataset, device)

    if not args.do_train and not args.do_predict:
        print("Nothing to do. Add --do_train and/or --do_predict.")


if __name__ == "__main__":
    main()

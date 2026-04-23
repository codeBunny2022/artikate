from __future__ import annotations
import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LABELS   = ["billing", "technical_issue", "feature_request", "complaint", "other"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MODEL_ID  = "distilbert-base-uncased"
MODEL_DIR = "./section3/model"
MAX_LEN   = 128


# dataset

class TicketDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer):
        self.examples  = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"].strip(),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(LABEL2ID[ex["label"]], dtype=torch.long),
        }


# trainer

class Trainer:
    def __init__(self, model_dir: str = MODEL_DIR, epochs: int = 5,
                 batch_size: int = 32, lr: float = 2e-5, seed: int = 42):
        self.model_dir  = Path(model_dir)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

    def train(self, data_path: str) -> None:
        with open(data_path) as f:
            all_data = json.load(f)

        train_data, val_data = self._stratified_split(all_data, val_ratio=0.15)
        logger.info("Train: %d | Val: %d", len(train_data), len(val_data))

        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_ID)
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(LABELS),
            id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(self.device)

        train_loader = DataLoader(TicketDataset(train_data, tokenizer),
                                  batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(TicketDataset(val_data, tokenizer),
                                  batch_size=self.batch_size)

        optimizer    = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps  = len(train_loader) * self.epochs
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * 0.1), total_steps)

        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler)
            val_acc, val_loss = self._eval_epoch(model, val_loader)
            logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                        epoch, self.epochs, train_loss, val_loss, val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                self.model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(self.model_dir))
                tokenizer.save_pretrained(str(self.model_dir))
                logger.info("Saved best model (val_acc=%.4f)", best_acc)

        logger.info("Training complete. Best val_acc: %.4f", best_acc)

    def _train_epoch(self, model, loader, optimizer, scheduler) -> float:
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
        return total / len(loader)

    def _eval_epoch(self, model, loader) -> tuple[float, float]:
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in loader:
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss_sum += outputs.loss.item()
                preds    = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])
        return correct / total, loss_sum / len(loader)

    @staticmethod
    def _stratified_split(data: list[dict], val_ratio: float):
        by_class: dict[str, list] = defaultdict(list)
        for ex in data:
            by_class[ex["label"]].append(ex)
        train, val = [], []
        for examples in by_class.values():
            n_val = max(1, int(len(examples) * val_ratio))
            val.extend(examples[:n_val])
            train.extend(examples[n_val:])
        return train, val


# inference

class TicketClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. Run: python -m section3.classifier train")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
        self.model     = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = torch.device("cpu")  # enforce CPU per spec
        self.model.to(self.device)
        logger.info("Classifier loaded from %s", model_dir)

    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        start = time.perf_counter()
        enc   = self.tokenizer(
            text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits[0]

        probs      = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_id    = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "label":      ID2LABEL[pred_id],
            "confidence": float(probs[pred_id]),
            "all_scores": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train")
    tp.add_argument("--data-path", default="section3/data/train.json")
    tp.add_argument("--epochs", type=int, default=5)

    pp = sub.add_parser("predict")
    pp.add_argument("--text", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        Trainer(epochs=args.epochs).train(args.data_path)
    elif args.cmd == "predict":
        clf    = TicketClassifier()
        result = clf.predict(args.text)
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Latency:    {result['latency_ms']:.1f}ms")

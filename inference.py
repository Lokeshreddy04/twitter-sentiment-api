import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ---------- Tokenization ----------

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
TOKEN_PATTERN = re.compile(
    r"URL|USER|[a-z]+(?:'[a-z]+)?|\d+|[:;=8][-^']?[)(/DPp]|[!?.,]+|[^\w\s]",
    re.IGNORECASE,
)


def normalize_tweet(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" URL ", text)
    text = MENTION_PATTERN.sub(" USER ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_tweet(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_tweet(text))


# ---------- Model ----------

class TwitterGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)
        features = self.dropout(hidden[-1])

        return self.classifier(features).squeeze(-1)


# ---------- Inference service ----------

class SentimentPredictor:
    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = Path(artifact_dir)

        with open(
            self.artifact_dir / "vocab.json",
            encoding="utf-8",
        ) as file:
            self.vocab = json.load(file)

        with open(
            self.artifact_dir / "tokenizer_config.json",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        checkpoint = torch.load(
            self.artifact_dir / "best_twitter_gru.pth",
            map_location="cpu",
            weights_only=True,
        )

        self.max_length = checkpoint["max_length"]
        self.unk_id = self.vocab["<UNK>"]

        self.model = TwitterGRU(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_size=checkpoint["hidden_size"],
            dropout=checkpoint["dropout"],
            padding_idx=checkpoint["padding_idx"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        tokens = tokenize_tweet(text)
        token_ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens
        ]

        token_ids = token_ids[-self.max_length:]

        if not token_ids:
            token_ids = [self.unk_id]

        return token_ids


    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Tweet text must be a non-empty string."
            )

        token_ids = self.encode(text)
        length = len(token_ids)

        padded_ids = token_ids + [
            self.vocab["<PAD>"]
        ] * (self.max_length - length)

        input_ids = torch.tensor(
            [padded_ids],
            dtype=torch.long,
        )
        lengths = torch.tensor([length], dtype=torch.long)

        self.model.eval()

        logit = self.model(input_ids, lengths)
        probability = torch.sigmoid(logit).item()

        return {
            "text": text,
            "sentiment": (
                "positive" if probability >= 0.5 else "negative"
            ),
            "positive_score": round(probability, 4),
            "negative_score": round(1 - probability, 4),
        }

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            raise ValueError("At least one tweet is required.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every tweet must contain non-whitespace text.")

        encoded = [self.encode(text) for text in texts]
        lengths = torch.tensor(
            [len(ids) for ids in encoded],
            dtype=torch.long,
    )

        padded = [
            ids + [self.vocab["<PAD>"]] * (self.max_length - len(ids))
            for ids in encoded
    ]

        input_ids = torch.tensor(padded, dtype=torch.long)
        self.model.eval()
        logits = self.model(input_ids, lengths)
        probabilities = torch.sigmoid(logits).tolist()

        results = []

        for text, probability in zip(texts, probabilities):
            results.append({
            "text": text,
            "sentiment": "positive" if probability >= 0.5 else "negative",
            "positive_score": round(probability, 4),
            "negative_score": round(1 - probability, 4),
        })

        return results


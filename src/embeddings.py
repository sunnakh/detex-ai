import time
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

start_time = time.time()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v5-text-small", trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v5-text-small", trust_remote_code=True, torch_dtype=dtype
).to(device)


def embeddings(texts: list[str], batch_size: int = 8) -> torch.Tensor:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1)

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)



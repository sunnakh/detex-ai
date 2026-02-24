import torch
import time
from transformers import AutoModel, AutoTokenizer


start = time.time()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(device)

print("Using device:", device)


texts = [
    "My order hasn't arrived yet and it's been two weeks.",
    "How do I reset my password?",
    "I'd like a refund for my recent purchase.",
    "Your product exceeded my expectations. Great job!",
]
classification_embeddings = model.encode(texts=texts, task="classification")

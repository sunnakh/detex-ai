import types
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer

m = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    model_kwargs={"dtype": torch.float32, "default_task": "retrieval"},
)

jina = m[0].model
backbone = jina.model  # PEFT MixedModel (backbone.layers via __getattr__)
layers = backbone.layers

# Freeze all, unfreeze last 4
for p in m.parameters():
    p.requires_grad = False
for layer in layers[-4:]:
    for p in layer.parameters():
        p.requires_grad = True
if hasattr(backbone, "norm"):
    for p in backbone.norm.parameters():
        p.requires_grad = True

trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")


def _jina_forward_with_grad(
    self,
    features: Dict[str, torch.Tensor],
    task: Optional[str] = None,
    truncate_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if task is None:
        task = self.default_task
    self.model.set_adapter(task)
    device = self.model.device

    def _compute(batch):
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state
        mask = batch.get("attention_mask")
        if mask is None:
            pooled = hidden[:, -1]
        else:
            seq_len = mask.sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), seq_len
            ]
        if truncate_dim is not None:
            pooled = pooled[:, :truncate_dim]
        return F.normalize(pooled, p=2, dim=-1)

    batch = {k: v.to(device) for k, v in features.items() if torch.is_tensor(v)}
    if self.training:
        self.model.train()
        features["sentence_embedding"] = _compute(batch)
    else:
        self.model.eval()
        with torch.no_grad():
            features["sentence_embedding"] = _compute(batch)
    return features


m[0].forward = types.MethodType(_jina_forward_with_grad, m[0])

# Test
m.train()
feat = m.tokenize(["hello world test"])
out = m[0](feat)
emb = out["sentence_embedding"]
print("embedding requires_grad:", emb.requires_grad)
print("embedding grad_fn:", emb.grad_fn)

# Test backward through a dummy loss
if emb.requires_grad:
    loss = emb.sum()
    loss.backward()
    print("backward() succeeded!")
else:
    print("FAIL: no grad on embedding")

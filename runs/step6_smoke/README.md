# Step 6 Smoke Test

Synthetic end-to-end invocation of the online trainer on CPU to verify the Step 6 objectives.

```bash
python - <<'PY'
import torch

from dvi.schedule import PiecewiseSchedule
from dvi.online_trainer import OnlineTrainer


class DemoModel(torch.nn.Module):
    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.base = torch.nn.Linear(d_model, vocab, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora = torch.nn.Parameter(torch.zeros(vocab, d_model))

    def drafter_logits_from_hk(self, hk: torch.Tensor) -> torch.Tensor:
        weight = self.base.weight + self.lora
        return hk @ weight.t()

    def dvi_trainable_params(self):
        return [self.lora]


torch.manual_seed(0)
model = DemoModel(d_model=3, vocab=5)
schedule = PiecewiseSchedule(warmup=2, kl0=1.0, klmin=0.1, pgmax=0.2)
trainer = OnlineTrainer(
    model,
    lr=5e-4,
    tau=1.5,
    schedule=schedule,
    max_ms=5,
    store_mode="topk",
    topk=3,
    vocab_size=5,
    device=torch.device("cpu"),
)

batch = {
    "hk": torch.randn(4, 3),
    "token": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
    "reward": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    "pos": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    "is_first_reject": torch.tensor([False, False, True, False]),
    "z_phi": None,
    "z_idx": torch.tensor([[0, 1, 2], [0, 1, 2], [1, 3, 4], [0, 2, 4]], dtype=torch.int32),
    "z_val": torch.randn(4, 3),
}
metrics = trainer.step(batch, global_step=0)
print({k: round(v, 6) for k, v in metrics.items()})
PY
```

Output:

```
{'loss': 1.081121, 'kd': 0.181985, 'ce': 1.576009, 'pg': -1.050672, 'kl': 0.202124, 'ms': 22.591352, 'acc_ratio': 0.5}
```

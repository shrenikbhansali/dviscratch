# Step 5 Smoke Test

- Buffer config: `capacity=8`, `d_model=4`, `vocab_size=16`, `store="topk"`, `topk=4`, `logits_dtype=torch.float16`, `device="cpu"`.
- Sample `stats()` output:
  ```python
  {'size': 5, 'capacity': 8, 'fill_frac': 0.625, 'hk_bytes': 64.0, 'meta_bytes': 88.0, 'logits_bytes': 192.0, 'total_bytes': 344.0}
  ```
- Tests: `pytest`

# Step 3 Smoke Check

## CLI used to attach the drafter head
```
python -m evaluation.inference_kangaroo \
  --model-id lmsys/vicuna-7b-v1.5 \
  --adapter-path none \
  --exitlayer 2 --threshold 0.6 --steps 6 \
  --dvi-attach-drafter true --dvi-lora-rank 8 --dvi-lora-alpha 8 \
  --temperature 0.0 --seed 123
```

## Trainable LoRA parameters
- `drafter_head.proj.A` with shape `[vocab_size, rank]`
- `drafter_head.proj.B` with shape `[rank, hidden_size]`
- Example for Vicuna-7B (vocab=32,000, hidden=4,096, rank=8): 288,768 trainable parameters total.

## Observations
- Greedy outputs with the drafter head attached match the baseline verifier-only run before any training.

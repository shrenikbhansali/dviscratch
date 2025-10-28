This repo is a copy of the Kangaroo paper's repo. We are modifying it to implement a DVI training approach. 

Here is the workplan:

Below is the **final, corrected, end-to-end implementation plan**. It integrates the two new corrections:

1. **Identity adapter bootstrap**: when `--adapter-mode none`, **derive** the adapter config from the **base model config** (no disk reads), **copy** weights from the base model to construct a true identity shallow branch, and **populate metadata** so no code attempts to reopen a missing checkpoint.

2. **Top-K KD/KL normalization**: when `store=topk`, compute **all KD/KL terms in gathered [B,K] space** with both the **verifier** and **drafter** distributions **renormalized over K** (no scattering to `[B,|V|]`). This is an **approximation**; we state it explicitly.

Everything else from earlier drafts is reconciled and restated here so you can implement PR-by-PR without ambiguity.

---

# Global constraints (apply to all steps)

* **Lossless**: with `--dvi-online false`, decoded text must match the base model’s greedy output for a given seed.
* **Frozen verifier/backbone**: never update or replace base weights used by the verifier path (including `head_model`).
* **Drafter ≠ Verifier**: **separate LoRA-backed drafter head**; verifier keeps using **`self.head_model`** unchanged.
* **No batching**: streaming, one example at a time.
* **Global step**: increment **once per verify/commit block**.

---

## STEP 0 — CLI & constructor foundations (PR-0)

### Goal

Make the adapter truly **optional** without file I/O, unify/validate flags, and add DVI CLI everywhere.

### Touch / Don’t touch

* **Touch**:
  `kangaroo/kangaroo_model.py` (constructor), `evaluation/inference_kangaroo.py` (driver), `kangaroo/cli_utils.py` (NEW).
* **Don’t touch**: draft/verify algorithm, commit rule, dynamic early exit.

### What to implement

#### 0.1 CLI (normalize flags)

Accept both legacy and canonical names; normalize:

```text
--model-id <hf_id_or_path>            # canonical (required)
--model-path <legacy>                 # if provided: must equal --model-id

--adapter-path <path|none>            # default 'none'
--adapter-mode {auto,none,load}       # default 'auto'

# DVI (plumbing only here; used later)
--dvi-online {true,false}
--dvi-batch-size <int>
--dvi-update-every <int>
--dvi-buffer-size <int>
--dvi-store {full,topk}
--dvi-topk <int>
--dvi-logits-dtype {float16,float32}
--dvi-tau <float>
--dvi-warmup-steps <int>
--dvi-pg-lambda-max <float>
--dvi-kl-lambda0 <float>
--dvi-kl-lambdamin <float>
--dvi-entropy-weight <float>
--max-online-train-ms <int>
--load-lora <path_or_empty>
--save-lora-every <int>
--save-lora-path <dir>
--seed <int>
```

Resolution logic:

* If `--model-path` present, must equal `--model-id`.
* Adapter mode:

  * `none` **or** `adapter-path='none'` → identity adapter path (no disk).
  * `load` **or** real path → load from checkpoint.
  * `auto` → if `adapter-path` provided use `load`, else `none`.

#### 0.2 KangarooModel: identity adapter bootstrap (no file I/O)

**Problem fixed**: constructor used to unconditionally read adapter files. New behavior:

```python
class KangarooModel(nn.Module):
    def __init__(self, *, model_id:str, adapter_mode:str, adapter_path:Optional[str], exit_layer:int, **kw):
        base_cfg = AutoConfig.from_pretrained(model_id)   # HF
        self._load_base_model(model_id)                   # base LM (verifier backbone)
        if adapter_mode == "load":
            self._load_adapter_from_path(adapter_path)
        else:  # 'none' path: build identity adapter from base
            self._build_identity_adapter_from_base(base_cfg, exit_layer)
```

**Design of the identity adapter** (exact requirements):

* **Derive adapter config**:
  `adapter_cfg = deepcopy(base_cfg)`; set metadata used downstream:

  * `adapter_cfg.hidden_size = base_cfg.hidden_size`
  * `adapter_cfg.intermediate_size = base_cfg.intermediate_size`
  * `adapter_cfg.num_hidden_layers = exit_layer`
  * `adapter_cfg.vocab_size = base_cfg.vocab_size`
  * Copy rotary/rope, norm eps, attention heads, kv heads, max position embeddings, etc.
  * Mark a field like `adapter_cfg.is_identity = True` to signal no Pytorch files are expected.

* **Construct shallow blocks by **copying weights** from the base model**:

  ```python
  def _build_identity_adapter_from_base(self, base_cfg, exit_layer:int) -> None:
      """
      Build a shallow stack (layers [0..exit_layer-1]) as a *real* module whose
      parameters are COPIES of the base model's corresponding layers.
      All dropouts set to 0; LayerNorm (RMS) eps identical; masks identical.
      """
      # Example (names are illustrative; adapt to the repo's module names):
      self.shallow_stack = ShallowStack(
          layers=[deepcopy_from(self.base_model.model.layers[i]) for i in range(exit_layer)]
      )
      # Freeze these copies to preserve losslessness (no grads on identity path):
      for p in self.shallow_stack.parameters(): p.requires_grad = False

      # Expose forward_early_stop consistent with downstream code:
      # forward_early_stop(x, past_kv, **kwargs) -> returns hk at split k
  ```

  *Rationale*: by cloning the **exact** base-layer weights (not random init), the shallow forward produces the **same activations** the verifier expects; downstream won’t try to reopen missing adapter files because `adapter_cfg` is fully populated.

* **Wire metadata**: ensure any code that reads `adapter_config.hidden_size`, etc., finds valid values; never touch filesystem for adapter when `mode=none`.

* **Outcome**: with `--adapter-mode none`, the model builds, shallow features are deterministic, and the program can run end-to-end **before** any LoRA head is attached.

### Deliverables & acceptance

* Running:

  ```bash
  python -m evaluation.inference_kangaroo \
    --model-id lmsys/vicuna-7b-v1.5 \
    --adapter-path none \
    --exitlayer 2 --threshold 0.6 --steps 6 --temperature 0.0 --seed 123
  ```

  works: no adapter disk reads; decode finishes; counters increment.

---

## STEP 1 — Baseline smoke runs (PR-1)

**Goal**: capture reference outputs/timings.

Commands (now valid with identity adapter):

```bash
python -m evaluation.inference_baseline \
  --model-id lmsys/vicuna-7b-v1.5 --temperature 0.0 --seed 123

python -m evaluation.inference_kangaroo \
  --model-id lmsys/vicuna-7b-v1.5 \
  --adapter-path none \
  --exitlayer 2 --threshold 0.6 --steps 6 \
  --temperature 0.0 --seed 123
```

**Deliverable**: `runs/baseline/README.md` (env, commands, sample tokens, runtime).

---

## STEP 2 — DVI scaffolding + **jsonl** ShareGPT streaming (PR-2)

**Goal**: Add DVI types and a true line-wise loader (no bulk `json.load`).

**Files**: `dvi/__init__.py`, `dvi/types.py`, `dvi/utils.py` (optional), `data/sharegpt_stream.py`.

**Type**:

```python
class DVITrainSample(NamedTuple):
    hk: "torch.Tensor"                  # [d_model] float16
    token: int
    reward: int                         # 1 accepted, 0 first reject
    pos: int                            # 1..m or m+1
    z_phi: Optional["torch.Tensor"]     # [V] if store=full
    z_idx: Optional["torch.Tensor"]     # [K] if store=topk
    z_val: Optional["torch.Tensor"]     # [K] if store=topk
    is_first_reject: bool
```

**Loader** (line-wise):

```python
def iter_sharegpt_jsonl(path:str, max_src_len:int=2048, max_tgt_len:int=256):
    """Yield (prompt_text, target_text) line-by-line (no batching)."""
```

**Acceptance**: memory flat while peeking; prints 3 pairs.

---

## STEP 3 — Separate **drafter head (LoRA)**; verifier stays on `head_model` (PR-3)

**Goal**: ensure LoRA never changes verifier logits.

**Files**: `kangaroo/drafter_head.py` (NEW), `kangaroo/kangaroo_model.py` (helpers).

**Drafter head**:

```python
class LoRALinear(nn.Module):
    # base weight/bias frozen (requires_grad=False)

class DrafterHead(nn.Module):
    def __init__(self, in_f:int, vocab_size:int, r:int=8, alpha:float=8.0):
        self.proj = LoRALinear(in_f, vocab_size, r=r, alpha=alpha, bias=False)
    def forward(self, hk): return self.proj(hk)        # [N,V]
    def lora_params(self): return self.proj.trainable_params()
```

**KangarooModel helpers** (use **existing `self.head_model`** as source):

```python
def attach_drafter_head(self, *, r:int, alpha:float):
    self.drafter_head = DrafterHead(self.config.hidden_size, self.config.vocab_size, r=r, alpha=alpha).to(self.device)
    # Seed drafter base weights from verifier's projection (head_model) WITHOUT touching head_model:
    self.drafter_head.proj.base.weight.data.copy_(self.head_model.weight.data)

def drafter_logits_from_hk(self, hk): return self.drafter_head(hk)
def dvi_trainable_params(self): return list(self.drafter_head.lora_params()) if hasattr(self, "drafter_head") else []
```

**Acceptance**:

* Verifier logits unchanged before/after attaching drafter head.
* `dvi_trainable_params()` returns only LoRA tensors.

---

## STEP 4 — Hidden-state capture (hk) at the split (PR-4)

**Goal**: persist **post-norm** shallow hidden for each drafted token.

**Contract**:

* Capture the exact tensor used to project to drafter logits (post-norm at layer `k`).
* Store as `float16` `[d_model]`.
* Provide:

  ```python
  def hk_state_of(i:int) -> "torch.Tensor": ...
  ```

**Acceptance**: debug flag prints count/bytes per block; bounded by buffer cap.

---

## STEP 5 — Ring buffer with **full vs top-K** storage (PR-5)

**Goal**: memory-bounded tuples; **never** rebuild dense tensors in top-K mode.

**API**:

```python
class DVIRingBuffer:
    def __init__(capacity:int, d_model:int, vocab_size:int,
                 store:str, topk:int, logits_dtype:torch.dtype, device:str)
    def push_full(hk, token, z_phi, reward, pos, is_first_reject)
    def push_topk(hk, token, z_phi, reward, pos, is_first_reject)   # computes topk and stores (idx,val)
    def sample(n:int) -> dict  # returns fields matching DVITrainSample
```

**Memory guidance (Vicuna-7B)**: hk ~8KB/slot; full logits ~64KB/slot; top-K=1024 ~6KB/slot.

**Acceptance**: capacity bound; sample shapes correct; no dense expansion in top-K path.

---

## STEP 6 — Online trainer (KD → CE+PG + KL) with **Top-K normalization** & time cap (PR-6)

**Goal**: implement DVI objective s.t. top-K mode **renormalizes** both drafter and verifier **in K-space**; enforce wall-clock budget.

**Files**: `dvi/online_trainer.py`, `dvi/schedule.py`.

**Schedule**:

```python
class PiecewiseSchedule:
    def __init__(self, warmup:int, kl0:float, klmin:float, pgmax:float):
        self.step=0; ...
    def weights(self) -> dict: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, s:dict) -> None: ...
```

**Trainer API**:

```python
class OnlineTrainer:
    def __init__(self, model, lr, tau, schedule, max_ms, store_mode, topk, vocab_size): ...
    def step(self, batch:dict, global_step:int) -> dict: ...
```

**Key computations (no dense in top-K)**

Let `draft_logits` be `[B, V]`. Let `(z_idx, z_val)` be the top-K verifier coordinates/values per sample. Define:

* **Drafter gathered logits**: `g = draft_logits.gather(1, z_idx)` → `[B,K]`

* **Renormalized drafter distribution on K**:
  ( p_k = \operatorname{softmax}(g, \text{dim}=-1) )
  ( \log p_k = \operatorname{log_softmax}(g, \text{dim}=-1) )

* **Verifier gathered distribution**:
  ( p_{\tau,k} = \operatorname{softmax}(z_val / \tau, \text{dim}=-1) )

* **KD (forward KL in K-space; **approximation**)**:
  [
  \mathcal{L}*{\mathrm{KD}}^{(K)} = \mathrm{KL}\Big(p*{\tau,k} ;\Big|; p_k\Big)
  = \sum_{j=1}^K p_{\tau,k}^{(j)} \big(\log p_{\tau,k}^{(j)} - \log p_k^{(j)}\big)
  ]

* **Reverse KL (stabilizer, K-space; **approximation**)**:
  [
  \mathcal{L}*{\mathrm{KL}}^{(K)} = \mathrm{KL}\Big(p_k ;\Big|; p*{\tau,k}\Big)
  = \sum_{j=1}^K p_k^{(j)} \big(\log p_k^{(j)} - \log p_{\tau,k}^{(j)}\big)
  ]

* **CE (accepted only)**: as before, use full-V `log_softmax(draft_logits)` gathered at action token id (this is exact for CE since action is one id).

* **PG (accepted + first reject)**: mask = `(reward==1) OR is_first_reject`; advantage `r - baseline`.

> **Note**: Top-K KLs are **approximations** of the full-V KLs projected onto the observed K coordinates. This is intentional to save memory; we **renormalize both sides** so gradients remain well-behaved.

**Time budget**: measure elapsed ms inside `step()`; return in stats.

**Acceptance**: tests confirm (a) no dense tensors built in top-K path, (b) KD decreases, (c) masks correct, (d) schedule save/load works, (e) report `ms`.

---

## STEP 7 — Integrate buffer + trainer in the loop (PR-7)

**Goal**: plug in logging, sampling, cadence, counters.

**Driver pseudocode**:

```python
# After verify/commit with longest agreeing prefix m for a draft of length k:
for i in range(1, m+1):
    buffer.push_*(hk=hk_state_of(i), token=draft_tokens[i-1],
                  z_phi=ver_logits[i-1], reward=1, pos=i, is_first_reject=False)
if m+1 <= len(draft_tokens):
    buffer.push_*(hk=hk_state_of(m+1), token=draft_tokens[m],
                  z_phi=ver_logits[m], reward=0, pos=m+1, is_first_reject=True)

global_step += 1

if args.dvi_online and (global_step % args.dvi_update_every == 0) and buffer.size >= args.dvi_batch_size:
    batch = buffer.sample(args.dvi_batch_size)
    stats = trainer.step(batch, global_step)  # stats include 'ms'
```

**Acceptance**:

* `--dvi-online false`: text == greedy verifier.
* `--dvi-online true`: rolling acceptance@1 increases; trainer logs `ms`.

---

## STEP 8 — LoRA checkpoint & resume (name-based filtering) (PR-8)

**Goal**: save/load **only** LoRA tensors + opt/schedule/baseline.

**API**:

```python
def lora_state_dict(model) -> dict:      # select params by name: endswith(".proj.A"), (".proj.B")
def load_lora_state_dict(model, sd:dict) -> None
def save_ckpt(path, model, opt, sched, ema_b) -> None
def load_ckpt(path, model, opt, sched) -> float
```

**Acceptance**: reproduces metrics on a fixed slice after reload.

---

## STEP 9 — Metrics (acceptance, MAT, wall-time) (PR-9)

**Goal**: log SD metrics.

**Output**:

* `runs/metrics/acceptance.csv`: `step, acc@1..@K, block_acc`
* `runs/metrics/speed.csv`: `step, wall_ms, speedup, drafter_ms, verifier_ms`

**Acceptance**: CSVs parse; summary prints periodically.

---

## STEP 10 — Benchmark (deployment mode) (PR-10)

**Goal**: evaluate with no online updates; load LoRA.

```bash
python -m evaluation.inference_kangaroo \
  --model-id lmsys/vicuna-7b-v1.5 \
  --adapter-path none \
  --exitlayer 2 --threshold 0.6 --steps 6 \
  --dvi-online false --load-lora runs/checkpoints/final.pt \
  --metrics-out runs/benchmark/metrics.csv --seed 123
```

**Acceptance**: better acceptance; non-worse wall-time vs baseline.

---

## STEP 11 — Ablations (PR-11)

**Modes**: `full`, `kd_only`, `ce_only`, `pg_only`, `kd+ce`, `kd+pg`.
Map to coefficient presets; identical cadence.

**Acceptance**: curves and end-state tables in `runs/ablations/`.

---

## STEP 12 — Tests & CI (PR-12)

**Must-cover**:

* Constructor identity: `--adapter-path none` **skips file I/O**; builds; decodes.
* Drafter isolation: verifier logits unchanged; only A/B trainable.
* Buffer: full & top-K push/sample; no dense expansion for top-K.
* Trainer: KD decreases; PG finite; **Top-K KLs renormalized**; schedule save/load.
* Lossless: with `--dvi-online false`, text == greedy verifier.
* Time budget: trainer returns `ms`; driver can warn if exceeded.

---

# Quick reference (defaults)

* Kangaroo: `--exitlayer 2 --threshold 0.6 --steps 6`
* LoRA: `r=8, alpha=8`
* Buffer/logits: `--dvi-store topk --dvi-topk 1024 --dvi-logits-dtype float16`
* Buffer/cadence: `--dvi-buffer-size 2048 --dvi-batch-size 64 --dvi-update-every 8`
* Schedule: `--dvi-tau 1.5 --dvi-warmup-steps 1000 --dvi-pg-lambda-max 0.2 --dvi-kl-lambda0 1.0 --dvi-kl-lambdamin 0.1`
* Time cap: `--max-online-train-ms 5`

---

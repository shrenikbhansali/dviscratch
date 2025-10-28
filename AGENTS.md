Top level note - you are using the latest version of the transformers package in your enviroment. The actual deployment enviroment will be using Transformers transformers==4.33.3. If you are searching the web for any documentation or reference, make sure you are looking at transformers==4.33.3. This should also hold for the implementation - you are implementing as if we had transformers==4.33.3, not the latest version. 

Also remember the existence of files like adapter.py - make sure you are using the already implemented methods, and not just re-implementing them and so on. 

---

## 0 · Mission (60-second brief)

We’re layering **DVI (Draft → Verify → Improve)**—a training-aware, *lossless* self-speculative approach—on top of **Kangaroo**’s self-speculative decoding. Start from a **vanilla Vicuna** base model (no pre-trained Kangaroo adapter), stream **ShareGPT** conversations **one at a time** (no batching), perform **online updates** to a **shallow drafter head (LoRA)** while keeping the **verifier/backbone frozen**, and then benchmark acceptance/speedup (Spec-Bench style).

---

## 1 · Repo origin & what’s already here

This repo is based on **Kangaroo** (“Lossless Self-Speculative Decoding via Double Early Exiting”). The upstream layout exposes the standard **inference** entry points (e.g., `evaluation/inference_kangaroo.py`) and canonical flags (`--exitlayer`, `--threshold`, `--steps`) and a timing helper in `evaluation/speed.py`.

**Quick background**  
Kangaroo trains a tiny **adapter** over a fixed shallow sub-network (drafter) and uses the remaining layers to verify; it also includes **dynamic early exiting** during drafting to avoid overspending on hard tokens. We reuse this geometry and add a training-aware loop.

---


## 3 · Immutable core (read before coding)

> The following must remain *functionally* intact to preserve **lossless** speculation and baseline comparability:
>
> - **Verifier/backbone weights and greedy verification sampler** — **do not train or alter** these.  
> - **Speculative decode contracts** — longest-agreeing-prefix commit rule; drafter proposes \(k\) tokens; verifier commits/overrides.  
> - **Kangaroo’s drafting policy controls** — keep `--exitlayer`, `--steps`, and **dynamic early exit** (`--threshold`) semantics unchanged.  
> - **KV-cache plumbing & data flow** — don’t introduce extra KV caches or change ownership between drafter and verifier.

If a PR must touch any of the above, it **must** include:
1) before/after **acceptance** and **wall-time speed** on the same machine/config;  
2) proof that decoded text equals the verifier’s greedy output when DVI is disabled (lossless contract).

---

## 4 · Big-picture: DVI over Kangaroo (what to implement)

**DVI** converts the verifier’s accept/reject outcomes into online supervision for a small **LoRA-parameterized drafter head** attached at the shallow split. Training uses a **KL→RL schedule** (online distillation warm-up, then reward-masked CE + small on-policy PG), while the verifier and backbone stay frozen. This preserves single-model, lossless deployment.

---

## 5 · Environment & data

- **Base model**: `lmsys/vicuna-7b-v1.5` (vanilla; accept license on first pull).
- **Dataset**: stream a ShareGPT-style JSON (no batching). Only one conversation/sample is processed at a time.
- **Benchmark**: Spec-Bench-style acceptance and speed metrics on the same device for apples-to-apples comparisons.

---

## 6 · Files & modules the agent will touch

- `kangaroo/` — add **LoRA** hooks for the **shallow drafter branch only** (e.g., `kangaroo/lora.py`), plus a helper to return **only** drafter-LoRA trainables.
- `evaluation/inference_kangaroo.py` — add **logging hooks** for accept/reject tuples and (behind a flag) invoke a tiny **online trainer** every M verify steps (strict time cap).
- `dvi/` (new) — add an **online trainer** (`online_trainer.py`) that implements: KD warm-up (KL to frozen verifier with temperature τ), then reward-masked CE + modest on-policy PG with EMA baseline and a decaying KL weight.

All other core codepaths (verification, greedy sampler, commit rule, early-exit policy) should be **left intact**.

---

## 7 · Online training signals (what to log)

At each speculative block (prefix at time \(t\)):

1. The drafter proposes a block \(\mathbf{y}_{t+1:t+k} = (y_{t+1},\dots,y_{t+k})\) from shallow state \(h_k\).
2. The verifier (frozen) greedily decodes \(\hat{\mathbf{y}}_{t+1:t+k}\) and yields **m = longest agreeing prefix**:
   \[
   m = \max \{ i \in \{0,\dots,k\}\;|\; y_{t+j}=\hat y_{t+j}\ \forall\, j\le i \}.
   \]
3. **Log tuples** for positions \(i=1..m\) (accepted, reward \(r_i=1\)) and the first mismatch \(i=m+1\) (rejected, reward \(r_{m+1}=0\)):
   \[
   \left(h^{(k)}_{t+i-1},\ a_{t+i}=y_{t+i},\ z^\phi_{t+i},\ r_i,\ i\right).
   \]
   Here \(z^\phi\) are verifier logits; **do not log beyond the first mismatch** (counterfactual region).

---

## 8 · DVI technical deep-dive (math & objectives)

### 8.1 Notation

- **Model & heads**  
  - Drafter distribution \(p_\theta(\cdot \mid h^{(k)}_{t})\) computed from shallow state \(h^{(k)}\) (layers \(0..k\)) with **LoRA-augmented** projections.  
  - Verifier distribution \(p_\phi(\cdot \mid h^{(L)}_{t})\) from full depth \(L\), **frozen** (parameters \(\phi\)).  
  - Let \(z^\phi \in \mathbb{R}^{|V|}\) be verifier logits; \(\tilde{p}_\phi^\tau = \mathrm{softmax}(z^\phi/\tau)\) be temperature-softened targets (τ>1).

- **Rewards & masks**  
  - For each drafted position \(i\) in a block: reward \(r_i \in \{0,1\}\), where \(r_i=1\) if the verifier **accepts** token \(a_{t+i}\); \(r_{m+1}=0\) for the **first reject**.  
  - Define acceptance mask \(\mathbb{1}_i^{\text{acc}}=\mathbb{1}[r_i=1]\) and **on-policy mask** \(\mathbb{1}_i^{\text{on}}=\mathbb{1}[i\le m+1]\).

### 8.2 Logged dataset (online)

From the stream we form mini-batches by sampling tuples:
\[
\mathcal{D}=\left\{\Big(h^{(k)}_{t+i-1},\, a_{t+i},\, z^\phi_{t+i},\, r_i\Big)\right\}\quad\text{with }i\le m+1.
\]
No tuples are collected for \(i>m+1\) to avoid counterfactual bias.

### 8.3 Objectives

We combine three main signals with a stabilizing KL term (and optional entropy bonus). Let \(\pi_\theta(\cdot)\equiv p_\theta(\cdot \mid h^{(k)})\).

**(A) Knowledge Distillation (KD) Warm-up**  
Minimize CE to the softened verifier targets:
\[
\mathcal{L}_{\mathrm{KD}}
= \mathbb{E}_{(h,z^\phi)}\big[-\, \tilde{p}_\phi^\tau(\cdot)^\top \log \pi_\theta(\cdot)\big]
= \mathbb{E}\big[\mathrm{KL}\big(\tilde{p}_\phi^\tau \,\|\, \pi_\theta\big)\big] + \text{const}.
\]
This aligns the drafter to the verifier early, improving calibration before RL.

**(B) Reward-masked Cross-Entropy (accepted only)**  
\[
\mathcal{L}_{\mathrm{CE}}
= \mathbb{E}_{(h,a,r)}\big[-\, \mathbb{1}^{\mathrm{acc}}\log \pi_\theta(a)\big].
\]
This is supervised learning on **accepted** tokens only (i.e., those that produced actual speedup).

**(C) On-policy Policy Gradient (accepted + first reject)**  
Let \(b\) be an EMA baseline (scalar or per-position). The REINFORCE part is:
\[
\mathcal{L}_{\mathrm{PG}}
= \mathbb{E}_{(h,a,r)}\big[-\, \mathbb{1}^{\mathrm{on}}\, (r - b)\, \log \pi_\theta(a)\big].
\]
Gradients push up probability mass on actions that yield acceptance (and push down on the first rejection) while using \(b\) to reduce variance.

**(D) Stabilizing KL to Verifier**  
A small reverse-KL (or CE to softened targets) keeps the drafter close to the verifier during RL:
\[
\mathcal{L}_{\mathrm{KL}}
= \mathbb{E}\big[\mathrm{KL}\big(\pi_\theta \,\|\, \tilde{p}_\phi^\tau\big)\big].
\]

**(E) Entropy Bonus (optional)**  
\[
\mathcal{L}_{\mathrm{ent}} = -\, \mathbb{E}\big[ \mathcal{H}(\pi_\theta) \big],
\]
encouraging mild exploration early on.

**Total objective with a KL→RL schedule**
\[
\boxed{\
\mathcal{L}(\theta)
= \lambda_{\mathrm{kd}}(t)\,\mathcal{L}_{\mathrm{KD}}
+ \lambda_{\mathrm{ce}}(t)\,\mathcal{L}_{\mathrm{CE}}
+ \lambda_{\mathrm{pg}}(t)\,\mathcal{L}_{\mathrm{PG}}
+ \lambda_{\mathrm{kl}}(t)\,\mathcal{L}_{\mathrm{KL}}
+ \lambda_{\mathrm{ent}}\,\mathcal{L}_{\mathrm{ent}}\
}
\]
with time-dependent weights:
- **Warm-up**: \(\lambda_{\mathrm{kd}}\!\uparrow\) (dominant), \(\lambda_{\mathrm{pg}}=0\), \(\lambda_{\mathrm{ce}}\) small, \(\lambda_{\mathrm{kl}}=\lambda_{\mathrm{kl},0}\).
- **After warm-up**: ramp **\(\lambda_{\mathrm{pg}}\!\uparrow\)**, **\(\lambda_{\mathrm{kl}}\!\downarrow\)**; keep \(\lambda_{\mathrm{ce}}\) moderate.

### 8.4 Gradients (sketch)

- KD/CE terms yield standard cross-entropy gradients on \(\theta\).
- PG term:
\[
\nabla_\theta \mathcal{L}_{\mathrm{PG}}
= -\, \mathbb{E}\big[\mathbb{1}^{\mathrm{on}}(r-b)\,\nabla_\theta \log \pi_\theta(a)\big].
\]
Clip \(\|\nabla_\theta\|\) (e.g., to 1.0) and optionally cap \(\|z^\phi\|\) via temperature τ.

### 8.5 Acceptance-aware credit assignment

- **Accepted positions** \((i\le m)\): supervise with CE and provide positive PG advantage if \(1-b>0\).  
- **First reject** \((i=m+1)\): PG penalty with advantage \((0-b)\), steering drafter away from confusable proposals.  
- **Beyond first reject**: no logging (counterfactual).  
- This aligns signals specifically with tokens that produce **speedup** (accepted) and those that **break** the draft (first reject).

### 8.6 LoRA parameterization (drafter-only)

For any targeted projection \(W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}\),
\[
W_\theta = W_0 + \alpha \cdot A B,\quad
A \in \mathbb{R}^{d_{\text{out}}\times r},\ B \in \mathbb{R}^{r\times d_{\text{in}}},
\]
with small rank \(r\) (e.g., 4–16) and scale \(\alpha\). **Only \(A,B\)** are trainable; all backbone/verifier weights remain frozen.

### 8.7 Scheduling & practical knobs

- **Temperature** τ ∈ [1.3, 1.7] for softened targets.  
- **Warm-up steps**: 1–2k verify steps (streamed).  
- **Update cadence**: one **micro-update** every M verify steps; budget **≤ `max_online_train_ms` ms** per update.  
- **Regularization**: modest \(\lambda_{\mathrm{kl}}\) (e.g., 0.1–1.0) decaying, entropy bonus optional.  
- **Optimizer**: AdamW on LoRA params only; weight decay small (e.g., 0.01); grad clip 1.0.

### 8.8 Complexity & latency

- **Drafting policy** unchanged (reuse Kangaroo’s dynamic early exit + `--steps`).  
- **Extra cost**: a tiny backward pass on LoRA params at cadence \(1/M\). Keep it bounded by wall-time cap.  
- **Memory**: ring buffer of tuples fits comfortably on GPU when storing compact states (e.g., projected \(h^{(k)}\) or cached logits).

---

## 9 · CLI flags (add these; values are suggestions)

```

# In evaluation/inference_kangaroo.py

--dvi_online true|false
--dvi_tau 1.5
--dvi_warmup_steps 1000
--dvi_lr 5e-5
--dvi_update_every 8
--dvi_buffer_size 2048
--dvi_pg_lambda_max 0.2
--dvi_kl_lambda0 1.0
--dvi_kl_lambdamin 0.1
--dvi_entropy_weight 0.0
--max_online_train_ms 5         # hard time cap per update

```

Keep Kangaroo’s inference knobs (unchanged contract), e.g., `--exitlayer 2 --threshold 0.6 --steps 6` for Vicuna-7B.

---

## 10 · How to run (agent smoke tests)

**Baseline (no DVI):**
```

python -m evaluation.inference_baseline 
--model-path lmsys/vicuna-7b-v1.5 --temperature 0.0

```

**Kangaroo (no DVI):**
```

python -m evaluation.inference_kangaroo 
--model-path lmsys/vicuna-7b-v1.5 
--exitlayer 2 --threshold 0.6 --steps 6 --dtype float16

```

**Kangaroo + DVI (online, streaming ShareGPT):**
```

python -m evaluation.inference_kangaroo 
--model-path lmsys/vicuna-7b-v1.5 
--exitlayer 2 --threshold 0.6 --steps 6 
--dvi_online true --dvi_tau 1.5 --dvi_warmup_steps 1000 
--dvi_lr 5e-5 --dvi_buffer_size 2048 --dvi_update_every 8 
--sharegpt_path /path/to/sharegpt.json --no_batching true

````

---

## 11 · Benchmarks & metrics

- **Acceptance**: report accepted@1…@k up to first mismatch (CTAR-style).  
- **Speed**: report **MAT** and **wall-time speedup** (Spec-Bench semantics).  
- **Harness**: use the project’s timing script and/or Spec-Bench harness; same device for comparisons.

---

## 12 · Guardrails (do **not** do these)

- ❌ **Don’t** batch training samples (DVI is **online**).  
- ❌ **Don’t** train or modify the **verifier/backbone** or change its sampler (must stay **lossless**).  
- ❌ **Don’t** alter Kangaroo’s **dynamic early exit** semantics or add extra KV caches.  
- ❌ **Don’t** backprop through tokens **after** the first mismatch (counterfactual).  

---

## 13 · Definition of Done (per PR)

Include in every PR:

1) **What changed & why** (short summary).  
2) **Exact commands run** and **stdout snippets** (acceptance & timing).  
3) **Correctness**: with `--dvi_online false`, outputs **match** greedy verifier.  
4) **Performance**: DVI shows **non-decreasing acceptance@1** over time on a 500-prompt slice.  
5) **Isolation**: only LoRA params are trainable; report trainable parameter count.  
6) **PR hygiene**: small diff, passing CI, clear PR description/summary.


---

## 15 · References for implementers (non-exhaustive)

- Kangaroo (self-speculative decoding via double early exiting; code & README).
- DVI (lossless self-speculative training; online KD→RL; accepted-token credit; LoRA drafter; tuple logging).
- Spec-Bench (acceptance, MAT/wall-time speedup).  
- Vicuna 7B v1.5 (base model).  
- ShareGPT datasets (streamed JSON conversations).


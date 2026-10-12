# Experiment log

One row per cell of the paper's results table. Fill a row the day you run it,
not the week after. The columns here map one-to-one onto Table 1 and the
ablation tables in `../../paper/main.tex`, so a complete log is a complete
paper.

Copy the blocks below for each run. Keep the run ID: it is the `--out-dir` you
passed, and it is how you find the trace JSONL again in week 7.

---

## Run register

| Run ID | Date | Config | Models | Harnesses | Items | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `results/20261012_smoke` | 2026-10-12 | default.yaml | scripted | H1-H4 | 24 | pipeline check, not a result |
|  |  |  |  |  |  |  |

---

## Table 1: main results (harness H1, independent + commutative-write pooled)

Fills the main results table of the paper. Replace every XX.X.

| Model | Params | Par. AST | FBR | OER | GDS | pass^8 | Turns | Lat. naive (s) | Lat. async (s) | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Instruct | 0.5B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen2.5-Instruct | 1.5B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen2.5-Instruct | 3B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen2.5-Instruct | 7B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen3 (no thinking) | 0.6B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen3 (no thinking) | 1.7B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Qwen3 (thinking) | 4B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Llama-3.2-Instruct | 1B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Llama-3.2-Instruct | 3B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Phi-3-mini-Instruct | 3.8B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| SmolLM2-Instruct | 1.7B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| xLAM-2-fc-r | 1B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| xLAM-2-fc-r | 3B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |
| Reference ceiling | 32B | XX.X | XX.X | XX.X | XX.X | XX.X | X.X | XX.X | XX.X |  |

---

## Figure 2a: fallback mode decomposition

| Model | M1 % | M2 % | M3 % | M4 % | M5 % | Run ID |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

These come straight from the `mode_shares` column of `summary.csv`.

---

## A1: harness condition

The H2 row is the one to watch. If Par. AST goes up while OER goes down, the
current benchmark incentive is actively harming end users.

| Model | Harness | Par. AST | FBR | OER | GDS | M3 % | M4 % | M5 % | Run ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | h1_native |  |  |  |  |  |  |  |  |
|  | h2_parallel_forced |  |  |  |  |  |  |  |  |
|  | h3_serial_sanctioned |  |  |  |  |  |  |  |  |
|  | h4_serial_repeat |  |  |  |  |  |  |  |  |

---

## A2: dependency regime

Expectation: OER near ceiling for the first two, a drop for order-sensitive
(the honest limit of the claim), and serialisation beating batching outright on
dependent calls.

| Model | Regime | Par. AST | OER | GDS | Notes | Run ID |
| --- | --- | --- | --- | --- | --- | --- |
|  | independent |  |  |  |  |  |
|  | commutative_write |  |  |  |  |  |
|  | order_sensitive |  |  |  |  |  |
|  | dependent |  |  |  |  |  |

---

## A3: fault injection (the falsification test)

Run this in week 3, not week 7. If small models reissue the failed call instead
of recovering, the graceful-degradation claim must be scoped to models above
whatever threshold shows up here. Record the threshold either way.

| Model | Trace type | Recovery Rate | Repeated-failed-call rate | Run ID |
| --- | --- | --- | --- | --- |
|  | serialised |  |  |  |
|  | batched |  |  |  |

---

## A4: output protocol and inference mode

| Model | Protocol | Thinking | Par. AST | OER | Run ID |
| --- | --- | --- | --- | --- | --- |
|  | native tool-call | off |  |  |  |
|  | instruction-following | off |  |  |  |
|  | native tool-call | on |  |  |  |

---

## A5: scaling k

The number worth reporting is the k at which M2 (complete) gives way to M3
(truncated), not the k at which fallback starts.

| Model | k | FBR | OER | M2 % | M3 % | Run ID |
| --- | --- | --- | --- | --- | --- | --- |
|  | 2 |  |  |  |  |  |
|  | 3 |  |  |  |  |  |
|  | 4 |  |  |  |  |  |
|  | 6 |  |  |  |  |  |

---

## Classifier validation

Do this before any run you intend to report. Hand-label 30 traces and compare.

| Date | Model | Traces labelled | Agreement with `classify.py` | Disagreements and cause |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

---

## Run notes

### Run ID: `results/...`

- **Date:**
- **Config / commit:**
- **What I expected:**
- **What happened:**
- **Is the surprise a bug or a finding?** (Check the trace JSONL by hand before
  answering. The first hypothesis is always a classifier or parser bug.)
- **What this changes about the next run:**
- **Paper section this feeds:**

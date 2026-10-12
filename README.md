# seqfall: is sequential fallback graceful degradation or failure?

Starter repository for Shivam Srivastava, Vizuara SLM Bootcamp.

## The question

Ask a small language model to make three independent tool calls in one turn and
it will often make one, wait for the result, make the next, and so on. Every
leaderboard scores that as a failure. The Berkeley Function Calling Leaderboard
compares the emitted call set to a reference set by abstract syntax tree match,
so a model that reaches exactly the right end state over three turns is counted
alongside a model that invented an API that does not exist.

Nobody has checked whether the user's request was actually satisfied.

That is the whole project. You are going to execute the sequential fallback and
the reference parallel plan in the same environment and compare the resulting
world state. If the states match, the field has been reporting a formatting
difference as a capability gap, and small-model agent harnesses should stop
fighting the fallback and start scheduling it. If they do not match, you will
know exactly which of the five failure modes is responsible and at what model
scale, which is still a better answer than the undifferentiated zero the status
quo assigns.

## Why this is publishable

The phenomenon is already catalogued. ToolScan (arXiv:2411.13547) names
*Insufficient API Calls* as the single most common tool-use error pattern. So
the field has measured how often this happens and has never measured whether it
matters. You are running the missing ablation on the most frequent error class
in tool use. That is a small, clean, defensible contribution, which is exactly
the right shape for a workshop paper in eight weeks.

Two papers to keep on your desk because they can hurt you:

- **W&D** (arXiv:2602.07359) shows parallel tool calling improves *accuracy*,
  not just latency, on deep-research tasks. Engage it directly; do not cite it
  in passing. Your scope is multi-call dispatch, not parallel search.
- **Feedback That Backfires** (arXiv:2608.23651) finds that models from 135M to
  1.7B *increase* their probability of repeating a tool call they just watched
  fail. Graceful serial degradation assumes the model learns from the
  intermediate observation. At the smallest sizes it may not. This is your
  pre-registered risk, and `configs/faults.yaml` is the experiment that tests
  it. Run it in week 3, not week 7.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you only want the evaluation pipeline and not local inference, the first five
lines of `requirements.txt` are enough. Torch and transformers are only needed
once you point the config at a real model.

## First three things to run

```bash
# 1. See the task suite. 240 items across 4 dependency regimes.
python -m src.tasks

# 2. End-to-end pipeline check. No GPU, no downloads, about two seconds.
bash scripts/smoke.sh

# 3. A real model. Uncomment qwen2.5-1.5b in configs/default.yaml first.
bash scripts/run_eval.sh configs/default.yaml
```

Step 2 uses the `scripted` backend, which is a stand-in that emits each fallback
mode with a fixed probability so you can exercise the classifier, the metrics
and the plots before trusting them with a real model. Its numbers are synthetic
by construction and resampled per harness condition, so do not read anything
into the harness differences it reports, and never put a scripted row in a
results table.

## Layout

```
src/sandbox.py    40 deterministic tools over an explicit WorldState, plus the
                  state digest that makes outcome equivalence provable
src/tasks.py      task generation across 4 dependency regimes and k in 2..6
src/models.py     scripted / transformers / OpenAI-compatible backends
src/harness.py    the 4 harness conditions, the rollout loop, tool-call parsing
src/classify.py   the five-mode taxonomy (M1..M5). The scientific core.
src/metrics.py    Par. AST, Fallback Rate, OER, GDS, pass^k, Recovery Rate
src/evaluate.py   entry point; writes traces JSONL and summary CSV
src/plots.py      regenerates the paper's Figure 2 from summary.csv
src/train.py      optional: DPO on serialisation quality, mined from your traces
configs/          default.yaml (main sweep), faults.yaml (the A3 risk test)
scripts/          smoke.sh, run_eval.sh, run_faults.sh
experiments/      log_template.md, one row per results-table row
```

## The four dependency regimes

This is the part that makes the paper honest. Sequential fallback is not
uniformly safe, and saying so precisely is more convincing than claiming it
always is.

| Regime | Calls | Expectation |
| --- | --- | --- |
| `independent` | read-only, commute | OER near its ceiling |
| `commutative_write` | writes to disjoint keys | OER near its ceiling |
| `order_sensitive` | writes to the same key | OER drops; the honest limit of the claim |
| `dependent` | call j consumes call i's output | serialisation should *beat* batching |

## The four harness conditions

| Condition | What it tests |
| --- | --- |
| `h1_native` | the status quo |
| `h2_parallel_forced` | is the fallback even suppressible by prompting? |
| `h3_serial_sanctioned` | the deliberate encouragement the title asks about |
| `h4_serial_repeat` | h3 plus a duplicated prompt, a zero-token reliability lever |

The sharpest possible finding lives in h2. If forcing the batched form converts
outcome-preserving M2 traces into hallucination-adjacent M5 traces, then the
current benchmark incentive is actively harming end users. Watch that cell.

## Metrics, in one line each

- **Par. AST**: the BFCL-style structural score. Reported so your numbers are
  comparable to published leaderboards.
- **Fallback Rate**: share of items where no parallel block was emitted.
- **OER**: among fallbacks, the share reaching a state identical to the
  reference plan. This is the headline number.
- **GDS**: OER discounted by turn and latency inflation. Report it over a sweep
  of alpha and beta, not one pair, so a reader with a different cost model can
  read off their own answer.
- **pass^8**: every one of eight repeats must succeed. Your claim is about
  dependability, not peak accuracy.
- **Recovery Rate**: under fault injection, the share that still reaches the
  reference state. Reported separately for serialised and batched traces.

## Notes on rigour

- The sandbox is deterministic on purpose. StableToolBench showed how much
  apparent tool-use failure is really API nondeterminism, and you cannot afford
  that noise in an eight-week project.
- `src/classify.py` is rule-based with no judge model. Every label must be
  reproducible from the trace and the state digest alone, or a reviewer will
  reasonably ask whether your taxonomy or your judge is doing the work.
- Report the latency twice, once on a naive runtime and once with concurrent
  per-turn execution. A production measurement of fused parallel calling
  (arXiv:2405.17438) found 4x more parallel calls and 40% token savings buying
  only a 12% end-to-end latency reduction. How much of "parallel is faster"
  belongs to the model and how much to the harness is an empirical question,
  and you can answer it.

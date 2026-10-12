# CLAUDE.md

Instructions for the coding agent working on this repository with Shivam.

## Project goal

Decide, empirically, whether a small language model's habit of falling back to
sequential tool calls when asked for parallel ones is a safe degradation path
or a genuine failure. The field scores it as failure because benchmarks match
the *shape* of the emitted call set (BFCL-style AST match) and never execute the
sequential run to see whether it reached the same end state. This repo executes
both and compares the resulting world state.

The deliverable is a workshop paper. Target: an EACL 2027 workshop, deadline
15 December 2026. Everything in this repo should be in service of filling in
Table 1 and Figure 2 of the draft in `../paper/`.

## The one invariant

**Outcome equivalence is decided by `WorldState.digest()`, never by a model.**
If you are ever tempted to add an LLM judge to decide whether two runs "did the
same thing", stop and raise it instead. The entire contribution rests on the
comparison being mechanical and reproducible. If a case genuinely cannot be
decided by state equality, that case belongs in a documented limitation, not in
a judge prompt.

## Layout and what each file owns

| File | Owns |
| --- | --- |
| `src/sandbox.py` | the 40 tools, `WorldState`, the digest, fault injection |
| `src/tasks.py` | task generation and the four dependency regimes |
| `src/models.py` | backends only; no parsing, no scoring |
| `src/harness.py` | the four harness conditions, the rollout loop, tool-call parsing |
| `src/classify.py` | the M1 to M5 taxonomy. Rule-based. No judge model. |
| `src/metrics.py` | Par. AST, FBR, OER, GDS, pass^k, Recovery Rate |
| `src/evaluate.py` | CLI, trace JSONL, summary CSV |
| `src/plots.py` | Figure 2 from `summary.csv` |
| `src/train.py` | optional DPO on serialisation quality |

Respect those boundaries. Parsing lives in `harness.py` and not in a backend,
because *how* a model mangles a parallel block is part of the measurement and a
backend that silently normalises it would destroy the data.

## Conventions

- Python 3.10+, standard library plus the pinned deps. Type hints on public
  functions. No framework additions without asking.
- Every experiment is reproducible from a config in `configs/` plus a seed.
  Nothing important is hard-coded in a script.
- Tools must stay pure functions of `WorldState`. No clocks, no randomness, no
  network. A nondeterministic tool silently invalidates every OER number.
- Write the raw trace JSONL before computing any metric. The qualitative
  appendix and all re-analysis come out of it, and re-running a sweep to
  recover a column you forgot is expensive.
- New metrics go in `metrics.py` with a one-line docstring saying which column
  of the paper's results table they fill.
- British or American spelling, but match the surrounding file. The code uses
  "serialised" to match the paper.

## Working agreement with Shivam

- He is doing this part-time over eight weeks alongside the Vizuara SLM
  bootcamp. Prefer a small, correct, runnable change over a large refactor.
- When he asks for a feature, ask which experiment it serves (A1 to A5 in the
  paper). If it serves none, it is probably scope creep.
- When a run produces a surprising number, the first hypothesis is a bug in the
  classifier or the parser, not a discovery. Check the trace JSONL by hand
  before anyone gets excited.
- Do not invent results. Placeholder cells in the paper stay "XX.X" until a
  real run fills them.

## Suggested next steps, roughly in order

1. **Validate the classifier by hand.** Run 30 traces from one real model,
   label M1 to M5 manually, and compare against `classify.py`. Report the
   agreement rate in the paper. If it is below about 95 percent, fix the rules
   before running anything at scale. This is the single highest-value hour in
   the project.
2. **Get one real model end to end.** Qwen2.5-1.5B-Instruct on H1, 240 items.
   Confirm the fallback actually happens at the rate the literature implies.
3. **Run the falsification test early** (`configs/faults.yaml`). If sub-2B
   models cannot recover from an observed tool error, the graceful-degradation
   claim has to be scoped, and it is much better to learn that in week 3 than
   in week 7.
4. **Fill the parameter ladder.** Qwen2.5 at 0.5B/1.5B/3B/7B under H1, then the
   other families. Use vLLM for 7B and above.
5. **Run the harness sweep** (H1 to H4). Watch the H2 cell: if forcing the
   parallel form lowers OER, that is the paper's sharpest result.
6. **Re-score the BFCL parallel splits** with the same classifier, so the
   methodological claim does not depend on our sandbox.
7. Only then consider `src/train.py`. Measurement first.

## Things that will waste a week if you let them

- Chasing a real public API. It will rate-limit, drift, and inject noise that
  you cannot distinguish from model behaviour. The deterministic sandbox exists
  for exactly this reason.
- Scaling to 900 items before the classifier is validated. You will have to
  throw the runs away.
- Fine-tuning before measuring. The whole point of the paper is the
  measurement; a DPO result without a trustworthy OER is unpublishable.
- Treating the scripted backend's numbers as results. It is a pipeline test.

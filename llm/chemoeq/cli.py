"""Commands for an experiment file. An experiment file ends with

    if __name__ == "__main__":
        cli(exp)

and is run as   python experiments/<name>.py <command> [options]

    prepare     build the segment table from the data (cached; this rebuilds it)
    prompt      print the prompt the model would get now
    prompts     the prompts a run sent and the replies (--iteration N, default the last; --index J)
    search      run the search (resumes from its checkpoint)
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

COMMANDS = ["prepare", "prompt", "prompts", "search"]


def cli(exp, argv=None):
    ap = argparse.ArgumentParser(prog=Path(exp.source or "experiment").name,
                                 description=f"experiment {exp.name}: {exp.signature}",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[2])
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--iterations", type=int, help="override the number of search iterations")
    ap.add_argument("--iteration", type=int, help="prompts: this iteration (default the last)")
    ap.add_argument("--index", type=int, help="prompts: only this prompt of the iteration (0, 1, ...)")
    a = ap.parse_args(argv)
    if a.iterations is not None:
        exp = dataclasses.replace(exp, iterations=a.iterations, source=exp.source)
    globals()["_" + a.command](exp, a)


def _prepare(exp, a):
    from . import data as D
    exp.data.ensure()
    print("check that these columns are right before trusting the output:")
    print(D.preview(exp.data) + "\n")
    p = D.prepare(exp.data)
    D.save(p, exp.prepared_path)
    df = p["df"]
    print(f"{len(df):,} segments, {df['track'].nunique():,} tracks, {len(p['conditions'])} conditions; "
          f"{df['has_prev'].mean():.0%} have a predecessor  ->  {exp.prepared_path}")
    counts = df.groupby("cond", observed=True).size()
    for c, v in p["conditions"].items():
        print(f"  {c:28s} files {v['n_files']:3d}  segments {counts.get(c, 0):8,d}")


def _search_setup(exp):
    from . import data as D
    from .experiment import input_summary
    from .search import feedback_data, search_arrays
    pr = D.load_for(exp)
    splits = D.split(pr["df"], exp)
    A = search_arrays(exp, splits)
    feedback = feedback_data(exp, splits, pr["conditions"])
    A["feedback"] = feedback[0] if feedback else None
    return pr, A, input_summary(exp, A["fit"], len(pr["conditions"]), feedback)


def _prompt(exp, a):
    from .candidate import evaluate
    pr, A, summary = _search_setup(exp)
    examples = [r for r in (evaluate(s, A["fit"], A["val"], exp.scoring(), diagnostics=True, diag_data=A["feedback"])
                            for s in exp.seeds) if r["ok"]]
    for m in exp.render(examples[: exp.examples_per_prompt], summary, pr["conditions"]):
        print(f"--- {m['role']} ---\n{m['content']}\n")


def _prompts(exp, a):
    import json
    path = exp.run_dir / "prompts.jsonl"
    if not path.exists():
        raise SystemExit(f"no {path}: prompts are saved from this version on (save_prompts=True)")
    recs = [json.loads(line) for line in open(path)]
    it = a.iteration if a.iteration is not None else max(r["iteration"] for r in recs)
    picked = [r for r in recs if r["iteration"] == it and (a.index is None or r["prompt"] == a.index)]
    if not picked:
        raise SystemExit(f"no saved prompts for iteration {it}"
                         + (f", index {a.index}" if a.index is not None else ""))
    for r in picked:
        print(f"===== iteration {r['iteration']}, prompt {r['prompt']}, island {r['island']} =====")
        for m in r["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print(f"--- reply ---\n{r['reply']}\n")


def _search(exp, a):
    from .search import describe_best, run
    best = run(exp)
    print("\n" + describe_best(best, exp.likelihood.outputs))



"""The variants of an experiment file (experiments/paper.py): a variant is a dict of Experiment fields, `settings` the
fields every variant shares. Repeat r of a variant uses split_seeds[r] and random_seed r (the LLM's sampling seeds).
Run r of variant v goes to <results>/<out>/<v>_r<r>/, <out> being the experiment file's name.

    variants_cli(VARIANTS, SETTINGS, repeats=30)      # at the end of the file

Commands:  list | repeats | <variant or all> <experiment command ...> [--repeat r]
"""
from __future__ import annotations

import sys
from pathlib import Path

from .cli import cli
from .experiment import Experiment

PER_REPEAT = {"search", "prompts"}   # others: repeat 0 only


def experiment(variants: dict, settings: dict, variant: str, repeat: int, out: str = "paper",
               source: str | None = None, **overrides) -> Experiment:
    if source is None:                              # the file that called this
        source = str(Path(sys._getframe(1).f_code.co_filename).resolve())
    kw = {**settings, **variants[variant], **overrides}
    split_seeds = kw.pop("split_seeds", None)       # one split_seed per repeat: files that share them share splits
    if split_seeds is not None:
        kw["split_seed"] = split_seeds[repeat]
    random_seeds = kw.pop("random_seeds", None)     # one random_seed per repeat, if not the repeat itself
    random_seed = repeat if random_seeds is None else random_seeds[repeat]
    kw.setdefault("out_dir", f"{out}/{variant}_r{repeat}")
    return Experiment(name=f"{out}_{variant}_r{repeat}", random_seed=random_seed, source=source, **kw)


def variants_cli(variants: dict, settings: dict, repeats: int = 3, out: str | None = None, argv=None):
    source = str(Path(sys._getframe(1).f_code.co_filename).resolve())     # the experiment file
    out = out or Path(source).stem
    _main(variants, settings, repeats, out, sys.argv[1:] if argv is None else argv, source)


def _main(variants, settings, repeats, out, argv, source):
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv and argv[0] == "list":
        for v, kw in variants.items():
            print(f"{v:16s} sets {', '.join(kw)}")
        return
    if argv and argv[0] == "repeats":
        print(repeats)
        return
    if not argv:
        print(__doc__)
        return
    which = list(variants) if argv[0] == "all" else [argv[0]]
    if not set(which) <= set(variants):
        sys.exit(f"unknown variant {argv[0]!r}; choose from: all, {', '.join(variants)}")
    cmd, only = list(argv[1:]), None
    if "--repeat" in cmd:                           # --repeat r: that repeat alone
        i = cmd.index("--repeat")
        try:
            only = int(cmd[i + 1])
        except (IndexError, ValueError):
            sys.exit("--repeat takes a number: the repeat (random_seed) to run")
        if not 0 <= only < repeats:
            sys.exit(f"--repeat {only}: this file has repeats 0 to {repeats - 1}")
        del cmd[i:i + 2]
    for v in which:
        rs = range(repeats) if cmd and cmd[0] in PER_REPEAT else [0]
        for r in (rs if only is None else [only]):
            print(f"\n=== {v}, repeat {r} ===")
            cli(experiment(variants, settings, v, r, out, source=source), cmd)

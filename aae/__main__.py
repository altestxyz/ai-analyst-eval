import runpy
import sys
from pathlib import Path

from aae import pipeline

HELP = """usage: python -m aae <command> [options]

stages     setup  taxi  contexts  smoke  pilot  freeze  run  shard  report  doctor
results    merge  table  rescore  verify  figures  headline  sql_match  actual_costs
analysis   report  charts  laptop_vs_frontier  judge  audit  ablation_report  infra_report  scale  demo  diagnose  quota  power

Run `python -m aae <command> --help` for the options of one command."""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(HELP)
        return 0
    name = argv[0]
    if name in pipeline.build_parser()._subparsers._group_actions[0].choices:
        return pipeline.main(argv)
    if (Path(__file__).parent / f"{name}.py").exists():
        sys.argv = [f"aae.{name}", *argv[1:]]
        runpy.run_module(f"aae.{name}", run_name="__main__")
        return 0
    print(f"unknown command {name!r}\n\n{HELP}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

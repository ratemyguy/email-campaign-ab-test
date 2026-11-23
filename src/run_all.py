"""ONE entry point: rebuild every output from the raw CSV.

    python -I src/run_all.py                  # full rebuild (about a minute)
    python -I src/run_all.py --skip-notebook  # everything except executing the notebook

Steps: download/verify raw data -> clean -> SQL checks -> randomization -> main effects ->
power -> regression adjustment -> heterogeneity -> targeting policy -> README numbers ->
notebook -> README check. Generated files in data/processed/, results/ and images/ are
deleted first, so nothing stale can survive a rebuild.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import config as cfg  # noqa: E402


def clean_outputs() -> None:
    for folder, pattern in [(cfg.PROCESSED_DIR, "*.csv"), (cfg.RESULTS, "*.csv"), (cfg.IMAGES, "*.png")]:
        folder.mkdir(parents=True, exist_ok=True)
        for f in folder.glob(pattern):
            f.unlink()


def main(argv: list[str]) -> None:
    import build_notebook
    import get_data
    import key_numbers
    import prepare
    import s01_randomization
    import s02_main_effects
    import s03_power
    import s04_regression_adjustment
    import s05_heterogeneity
    import s06_targeting_policy
    import sql_checks
    import verify_readme

    steps = [
        ("clean outputs", clean_outputs),
        ("get data", get_data.main),
        ("prepare", prepare.main),
        ("SQL checks", sql_checks.main),
        ("1 randomization", s01_randomization.main),
        ("2 main effects", s02_main_effects.main),
        ("3 power", s03_power.main),
        ("4 regression adjustment", s04_regression_adjustment.main),
        ("5 heterogeneity", s05_heterogeneity.main),
        ("6 targeting policy", s06_targeting_policy.main),
        ("key numbers", key_numbers.main),
    ]
    if "--skip-notebook" not in argv:
        steps.append(("notebook", build_notebook.main))
    steps.append(("README check", verify_readme.main))

    t_all = time.time()
    for name, fn in steps:
        t0 = time.time()
        print(f"\n== {name} ==")
        fn()
        print(f"   ({time.time() - t0:.1f}s)")
    n_csv = len(list(cfg.RESULTS.glob("*.csv")))
    n_png = len(list(cfg.IMAGES.glob("*.png")))
    print(f"\nDone in {time.time() - t_all:.0f}s: {n_csv} CSVs in results/, {n_png} charts in images/.")


if __name__ == "__main__":
    main(sys.argv[1:])

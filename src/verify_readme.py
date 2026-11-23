"""Check every number in README.md against results/key_numbers.csv.

* Code blocks, inline code, link targets, list/heading numbering and a short list of
  non-result tokens (years, version numbers, method names) are set aside first;
  each allowlisted token carries its reason.
* Every remaining number must appear in the display text of some entry in
  results/key_numbers.csv, which src/key_numbers.py builds from results/*.csv.
* Images and results files linked from the README must exist.

Writes results/readme_number_check.csv and exits non-zero on any unmatched number.

    python -I src/verify_readme.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402

NUM = re.compile(
    r"(?<![A-Za-z0-9_.,$])(?P<sign>[+\u2212-](?=\$?\d))?"
    r"(?P<num>\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\d+(?:\.\d+)?)"
)

# Phrases whose digits are not results (checked and removed before tokenizing).
ALLOW_PHRASES = {
    "March 20, 2008": "publication date of the author's post",
    "dated March 2008": "date of the source file",
    "from 2008": "year the dataset was published",
    "Lin (2013)": "citation year (Lin 2013, regression adjustment)",
    "SHA-256": "name of the checksum algorithm",
    "HC2": "name of the robust standard-error type",
    "Python 3.12": "Python version used",
    "python3.12": "Python version used",
    "50/50": "train/holdout split design",
    "0.1 threshold": "conventional |SMD| balance threshold",
    "95th percentile": "name of a percentile",
}


def normalize(token: str) -> str:
    return token.replace("$", "").replace(",", "")


def sign_of(sign: str | None) -> str:
    return "-" if sign in ("-", "\u2212") else "+"


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # fenced code
    text = re.sub(r"`[^`]*`", " ", text)                          # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)                      # link targets
    text = re.sub(r"^\s*#+\s*\d+\.\s", "", text, flags=re.M)      # numbered headings
    text = re.sub(r"^\s*\d+\.\s", "", text, flags=re.M)           # ordered-list markers
    return text


def main() -> pd.DataFrame:
    readme = cfg.README.read_text(encoding="utf-8")
    keys = pd.read_csv(cfg.RESULTS / "key_numbers.csv", keep_default_na=False)
    known: dict[str, list[tuple[str, str]]] = {}  # number -> [(sign, key)]
    for _, r in keys.iterrows():
        for m in NUM.finditer(str(r["display"])):
            known.setdefault(normalize(m.group("num")), []).append((sign_of(m.group("sign")), r["key"]))

    text = strip_markdown(readme)
    rows = []
    for phrase, reason in ALLOW_PHRASES.items():
        if phrase in text:
            rows.append({"token": phrase, "status": "allowlisted", "matched_keys": "", "reason": reason, "context": ""})
            text = text.replace(phrase, " ")

    for m in NUM.finditer(text):
        tok = m.group(0)
        ctx = text[max(0, m.start() - 45): m.end() + 25].replace("\n", " ")
        cands = known.get(normalize(m.group("num")), [])
        if m.group("sign"):  # a signed README number must match a key with the same sign
            cands = [c for c in cands if c[0] == sign_of(m.group("sign"))]
        hit = sorted({k for _, k in cands})
        rows.append(
            {
                "token": tok,
                "status": "matched" if hit else "UNMATCHED",
                "matched_keys": ";".join(hit),
                "reason": "",
                "context": ctx,
            }
        )
    out = pd.DataFrame(rows)

    out.to_csv(cfg.RESULTS / "readme_number_check.csv", index=False)
    missing_files = [
        link for link in re.findall(r"\]\(((?:images|results)/[^)]+)\)", readme) if not (cfg.ROOT / link).exists()
    ]
    unmatched = out[out["status"] == "UNMATCHED"]
    n_num = int((out["status"] != "allowlisted").sum())
    print(
        f"README check: {n_num} numbers, {n_num - len(unmatched)} matched to results/key_numbers.csv, "
        f"{len(unmatched)} unmatched; {int((out['status'] == 'allowlisted').sum())} allowlisted phrases; "
        f"{len(missing_files)} missing linked files"
    )
    if len(unmatched) or missing_files:
        print(unmatched[["token", "context"]].to_string())
        print("Missing files:", missing_files)
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
update_paper.py -- Read new experiment results and patch ijcai26.tex macros.

Usage:
  python3 update_paper.py                          # auto-find results
  python3 update_paper.py --results-dir ../results # explicit results dir
"""
import json, re, os, sys, argparse, subprocess
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def pct(k, n): return round(k / n * 100) if n else 0
def frac(k, n): return f"{k}/{n}"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def patch_macro(tex, name, value):
    """Replace \\newcommand{\\NAME}{old} with new value."""
    pat = r'(\\newcommand\{\\' + re.escape(name) + r'\}\{)[^}]*(})'
    new = r'\g<1>' + str(value) + r'\g<2>'
    result = re.sub(pat, new, tex)
    if result == tex:
        print(f"  WARNING: macro \\{name} not found in tex")
    else:
        print(f"  Updated \\{name} → {value}")
    return result

# ── ZebraLogicBench ──────────────────────────────────────────────────────────

def update_zlb(tex, path):
    d = load_json(path)
    s = d["summary"]
    n = s["direct"]["total"]
    print(f"\nZebraLogicBench: n={n}")

    kd  = s["direct"]["correct"];   tex = patch_macro(tex, "ZLDT",    frac(kd, n))
    kc  = s["cot"]["correct"];       tex = patch_macro(tex, "ZLCT",    frac(kc, n))
    kss = s["symstep"]["correct"];   tex = patch_macro(tex, "ZLSST",   frac(kss, n))
    ksg = s["symstep_g"]["correct"]; tex = patch_macro(tex, "ZLSGST",  frac(ksg, n))

    tex = patch_macro(tex, "ZLDpct",   f"{pct(kd,n)}\\%")
    tex = patch_macro(tex, "ZLCpct",   f"{pct(kc,n)}\\%")
    tex = patch_macro(tex, "ZLSSpct",  f"{pct(kss,n)}\\%")
    tex = patch_macro(tex, "ZLSGSpct", f"{pct(ksg,n)}\\%")

    # Update n in macro comment and inline references
    old_n_comment = "35 puzzles"
    new_n_comment = f"{n} puzzles"
    print(f"  n: {old_n_comment} → {new_n_comment}")

    # Update ZLB size table if present
    per = d.get("per_size", {})
    if per:
        print(f"  Size breakdown: {per}")

    return tex, n, kd, kc, kss, ksg

# ── AR-LSAT ──────────────────────────────────────────────────────────────────

def update_lsat(tex, path):
    d = load_json(path)
    s = d["summary"]
    n = s["direct"]["total"]
    print(f"\nAR-LSAT: n={n}")

    kd  = s["direct"]["correct"];    tex = patch_macro(tex, "LSATDtot",   frac(kd, n))
    kc  = s["cot"]["correct"];        tex = patch_macro(tex, "LSATCtot",   frac(kc, n))
    kss = s["symstep"]["correct"];    tex = patch_macro(tex, "LSATSStot",  frac(kss, n))
    ksg = s["symstep_g"]["correct"];  tex = patch_macro(tex, "LSATSGStot", frac(ksg, n))

    tex = patch_macro(tex, "LSATDpct",   f"{pct(kd,n)}\\%")
    tex = patch_macro(tex, "LSATCpct",   f"{pct(kc,n)}\\%")
    tex = patch_macro(tex, "LSATSSpct",  f"{pct(kss,n)}\\%")
    tex = patch_macro(tex, "LSATSGSpct", f"{pct(ksg,n)}\\%")

    return tex, n, kd, kc, kss, ksg

# ── LGP-14 multi-run ─────────────────────────────────────────────────────────

def report_lgp14_multirun(path):
    d = load_json(path)
    n_runs = d["n_runs"]
    stats = d["statistics"]
    print(f"\nLGP-14 multi-run (N={n_runs}):")
    for m, s in stats.items():
        print(f"  {m:<14} mean={s['mean_acc_pct']:.1f}% std={s['std_acc_pct']:.1f}%")
    return d

# ── ZLB table update (per-size rows) ─────────────────────────────────────────

def compute_per_size(path):
    """Compute per-size (2x3, 2x4, 3x3, 3x4) stats from per_puzzle list."""
    from collections import defaultdict
    d = load_json(path)
    per = d.get("per_puzzle", [])
    by_size = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in per:
        size = p.get("size", p.get("difficulty", "?"))
        for m in ["direct", "cot", "symstep", "symstep_g"]:
            if m in p:
                v = p[m]
                correct = v["correct"] if isinstance(v, dict) else int(v)
                by_size[size][m][0] += correct
                by_size[size][m][1] += 1
    return {s: {m: tuple(v) for m, v in ms.items()} for s, ms in by_size.items()}

def update_zlb_table(tex, path):
    """Rewrite per-size cell values in the ZebraLogicBench table."""
    per_size = compute_per_size(path)
    sizes = ["2x3", "2x4", "3x3", "3x4"]
    methods = ["direct", "cot", "symstep", "symstep_g"]

    for size in sizes:
        if size not in per_size:
            print(f"  WARNING: size {size} not in results")
            continue
        sz_tag = size.replace("x", r"{\\times}")
        for m in methods:
            k, n = per_size[size].get(m, (0, 0))
            print(f"  {size} {m}: {k}/{n}")

    # Rebuild per-size rows in table
    # Format: Direct & 0/9 & 0/9 & 0/11 & 0/6 & \ZLDT~(\ZLDpct)\\
    method_labels = {
        "direct":    "Direct",
        "cot":       "CoT",
        "symstep":   "SymStep",
        "symstep_g": r"\\textbf{SymStep+G}",
    }
    macro_map = {
        "direct":    (r"\\ZLDT", r"\\ZLDpct"),
        "cot":       (r"\\ZLCT", r"\\ZLCpct"),
        "symstep":   (r"\\ZLSST", r"\\ZLSSpct"),
        "symstep_g": (r"\\ZLSGST", r"\\ZLSGSpct"),
    }

    for m in methods:
        cells = []
        for size in sizes:
            k, n = per_size.get(size, {}).get(m, (0, 0))
            cells.append(frac(k, n))
        tot_macro, pct_macro = macro_map[m]
        if m == "symstep_g":
            new_row = (f"\\textbf{{SymStep+G}} & " +
                       " & ".join(cells) +
                       f" & \\textbf{{\\ZLSGST~(\\ZLSGSpct)}}\\\\")
            old_pat = (r"\\textbf\{SymStep\+G\}.*?\\\\")
        else:
            label = {"direct": "Direct", "cot": "CoT", "symstep": "SymStep"}[m]
            new_row = (f"{label}             & " +
                       " & ".join(cells) +
                       f" & {tot_macro}~({pct_macro})\\\\")
            old_pat = rf"{label}\s+&.*?\\\\"

        new_tex = re.sub(old_pat, new_row, tex, flags=re.DOTALL)
        if new_tex != tex:
            print(f"  Updated ZLB table row: {label if m != 'symstep_g' else 'SymStep+G'}")
            tex = new_tex
        else:
            print(f"  WARNING: Could not update ZLB table row for {m}")

    return tex

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Directory with result JSON files")
    parser.add_argument("--tex", default="../paper/ijcai26.tex",
                        help="Path to ijcai26.tex")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change, don't write")
    args = parser.parse_args()

    # Find results
    if args.results_dir:
        rdir = Path(args.results_dir)
    else:
        # Try ../results first, then experiments/ itself
        for candidate in ["../results", "."]:
            p = Path(candidate)
            if (p / "zebralogic_results.json").exists():
                rdir = p
                break
        else:
            print("ERROR: Could not find results directory. Pass --results-dir.")
            sys.exit(1)

    print(f"Results dir: {rdir.resolve()}")
    tex_path = Path(args.tex)
    print(f"TeX file:    {tex_path.resolve()}")

    tex = tex_path.read_text()

    changed = False

    # ZebraLogicBench
    zlb_path = rdir / "zebralogic_results.json"
    if zlb_path.exists():
        tex, n_zlb, kd, kc, kss, ksg = update_zlb(tex, zlb_path)
        update_zlb_table(tex, zlb_path)
        changed = True
        print(f"  SymStep+G on ZLB: {ksg}/{n_zlb} ({pct(ksg,n_zlb)}%)")
        if pct(ksg, n_zlb) < 90:
            print(f"  ⚠️  WARNING: SymStep+G accuracy dropped below 90% on ZLB!")
    else:
        print(f"\nSkipping ZLB update (not found: {zlb_path})")

    # AR-LSAT
    lsat_path = rdir / "lsat_results.json"
    if lsat_path.exists():
        tex, n_lsat, kd, kc, kss, ksg = update_lsat(tex, lsat_path)
        changed = True
        print(f"  SymStep+G on LSAT: {ksg}/{n_lsat} ({pct(ksg,n_lsat)}%)")
    else:
        print(f"\nSkipping LSAT update (not found: {lsat_path})")

    # LGP-14 multi-run (report only, no macro update needed)
    lgp_path = rdir / "lgp14_multirun.json"
    if lgp_path.exists():
        report_lgp14_multirun(lgp_path)
    else:
        print(f"\nSkipping LGP-14 multirun report (not found: {lgp_path})")

    if not changed:
        print("\nNo results found to update.")
        sys.exit(0)

    if args.dry_run:
        print("\nDry run — not writing.")
        return

    tex_path.write_text(tex)
    print(f"\nWrote updated tex to {tex_path}")

    # Recompile
    paper_dir = tex_path.parent
    print("\nRecompiling...")
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=paper_dir, capture_output=True, text=True
        )
        if r.returncode != 0:
            print("pdflatex ERROR:", r.stdout[-500:])
    print("Done.")

if __name__ == "__main__":
    main()

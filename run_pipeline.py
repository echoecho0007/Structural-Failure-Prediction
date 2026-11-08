"""Run the full pipeline end-to-end (data -> training -> evaluation -> sensitivity -> figures -> tests).

Usage:
    python run_pipeline.py                # full run (~7-10 min on 2 CPU cores)
    python run_pipeline.py --backend sklearn   # force the scikit-learn fallbacks
    python run_pipeline.py --fast         # smoke test with tiny grids (results NOT representative)
    python run_pipeline.py --report       # also rebuild report.pdf (needs pdflatex)
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="auto", choices=["auto", "sklearn", "native"])
ap.add_argument("--fast", action="store_true")
ap.add_argument("--report", action="store_true")
args = ap.parse_args()

steps = [
    ["-m", "src.data_generation"],
    ["-m", "src.train_models", "--backend", args.backend] + (["--fast"] if args.fast else []),
    ["-m", "src.evaluation"] + (["--skip-ood"] if args.fast else []),
    ["-m", "src.sensitivity_analysis"],
    ["-m", "src.visualization"],
    ["-m", "pytest", "-q", "tests"],
]
if args.report:
    steps.append(["report/build_report.py"])

t0 = time.time()
for s in steps:
    print(f"\n=== python {' '.join(s)} ===", flush=True)
    r = subprocess.run([sys.executable, *s])
    if r.returncode != 0:
        sys.exit(f"Step failed: {' '.join(s)}")
print(f"\nPipeline completed in {time.time() - t0:.0f}s")

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "verify": [
        "00_setup_and_data_check.ipynb",
        "01_appliance_40ms_contract.ipynb",
        "02_reproduce_baselines.ipynb",
        "05_verify_claims_and_plots.ipynb",
    ],
    "smoke": [
        "00_setup_and_data_check.ipynb",
        "01_appliance_40ms_contract.ipynb",
        "02_reproduce_baselines.ipynb",
        "03_retrain_dwellmlp.ipynb",
        "04_retrain_dwellobserver_t.ipynb",
        "05_verify_claims_and_plots.ipynb",
    ],
    "paper": [
        "00_setup_and_data_check.ipynb",
        "01_appliance_40ms_contract.ipynb",
        "02_reproduce_baselines.ipynb",
        "03_retrain_dwellmlp.ipynb",
        "04_retrain_dwellobserver_t.ipynb",
        "05_verify_claims_and_plots.ipynb",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMPERE publication notebooks.")
    parser.add_argument("--mode", choices=sorted(NOTEBOOKS), default="verify")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument(
        "--save-executed",
        action="store_true",
        help="Save executed notebook outputs in place when nbclient is available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback when a notebook fails.",
    )
    return parser.parse_args()


def _execute_with_nbclient(path: Path, *, save: bool) -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    if save:
        nbformat.write(notebook, path)


def _execute_fallback(path: Path, *, save: bool) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
    }
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            print(f"[fallback] cell {index}: {path.name}")
            cell["execution_count"] = index
            cell["outputs"] = []
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(source, str(path), "exec"), namespace)
            finally:
                if stdout.getvalue():
                    print(stdout.getvalue(), end="")
                    cell["outputs"].append(
                        {
                            "name": "stdout",
                            "output_type": "stream",
                            "text": stdout.getvalue().splitlines(keepends=True),
                        }
                    )
                if stderr.getvalue():
                    print(stderr.getvalue(), end="", file=sys.stderr)
                    cell["outputs"].append(
                        {
                            "name": "stderr",
                            "output_type": "stream",
                            "text": stderr.getvalue().splitlines(keepends=True),
                        }
                    )
    finally:
        os.chdir(old_cwd)
        if save:
            path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def execute_notebook(path: Path, *, save: bool) -> None:
    try:
        _execute_with_nbclient(path, save=save)
    except ModuleNotFoundError:
        print("[verify] nbformat/nbclient not installed; using stdlib fallback executor.")
        _execute_fallback(path, save=save)


def main() -> int:
    args = parse_args()
    env = os.environ
    env["PYTHONPATH"] = str(ROOT / "src")
    env["AMPERE_RUN_MODE"] = args.mode
    env["AMPERE_DATA_DIR"] = args.data_dir
    env["AMPERE_OUTPUT_DIR"] = args.output_dir
    env.setdefault("MPLBACKEND", "Agg")

    os.chdir(ROOT)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from ampere_public.publication import DataVerificationError, verify_data_files

    try:
        verify_data_files(args.data_dir)
    except DataVerificationError as exc:
        print("\n[verify] data preflight failed.")
        print(str(exc))
        return 1
    print("[verify] data preflight passed.")

    for name in NOTEBOOKS[args.mode]:
        path = ROOT / "notebooks" / name
        print(f"\n[verify] running {path.relative_to(ROOT)}")
        try:
            execute_notebook(path, save=args.save_executed)
        except Exception as exc:
            print(f"\n[verify] FAILED: {path.name}")
            print(str(exc))
            if args.debug:
                traceback.print_exc()
            return 1
    print("\n[verify] all requested notebooks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

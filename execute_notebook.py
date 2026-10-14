"""Execute main.ipynb from a fresh kernel using this Python environment."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    # Select this interpreter without installing a global/user kernelspec.
    local = ROOT / ".work/jupyter"
    kernel = local / "kernels/phase1-validated"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                                   "display_name": "PS-O1 validated environment", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(local) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(local / "runtime")
    os.environ["IPYTHONDIR"] = str(local / "ipython")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".work/matplotlib")
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(ROOT / "main.ipynb", as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="phase1-validated", resources={"metadata": {"path": str(ROOT)}}).execute()
    # Keep the portable default kernelspec in the submitted notebook.
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, ROOT / "main.ipynb")
    print("PASS: main.ipynb executed from a fresh kernel; all outputs saved.")


if __name__ == "__main__":
    main()

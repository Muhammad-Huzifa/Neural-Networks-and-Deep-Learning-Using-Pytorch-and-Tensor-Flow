"""Check active notebook structure and Python syntax without executing models."""
import argparse
import ast
import json
from pathlib import Path

def check(root):
    count = 0
    errors = []
    for path in sorted(root.rglob("*.ipynb")):
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
            if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
                raise ValueError("Expected a version-4 notebook")
            for index, cell in enumerate(nb["cells"], 1):
                if cell.get("cell_type") not in {"markdown", "code", "raw"}:
                    raise ValueError(f"Unknown cell type at cell {index}")
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell.get("source", []))
                if source.lstrip().startswith("%%"):
                    continue
                source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
                ast.parse(source, filename=f"{path}:cell-{index}")
            count += 1
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise SystemExit("\n".join(errors))
    if count == 0:
        raise SystemExit(f"No notebooks found under {root}")
    print(f"Checked {count} notebooks: JSON and Python syntax passed; no model execution performed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks")
    check(parser.parse_args().root)

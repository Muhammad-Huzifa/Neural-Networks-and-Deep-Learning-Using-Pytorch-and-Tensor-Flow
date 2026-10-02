"""Execute the self-contained lesson manifest in fresh kernels, keeping source outputs clean."""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


class CurrentInterpreterSpecs(KernelSpecManager):
    """Run the interpreter that launched this command, including its virtual environment."""

    def get_kernel_spec(self, kernel_name):
        if kernel_name == 'python3':
            return KernelSpec(
                argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                display_name='Python 3', language='python',
                env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'},
            )
        return super().get_kernel_spec(kernel_name)


def execute(manifest, only=None, report=None, output_dir=None):
    class LessonKernelManager(AsyncKernelManager):
        def __init__(self, *args, **kwargs):
            kwargs['kernel_spec_manager'] = CurrentInterpreterSpecs()
            super().__init__(*args, **kwargs)

    entries = json.loads(manifest.read_text(encoding='utf-8'))
    paths = [entry['path'] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths in lesson manifest')
    selected = paths if not only else only
    unknown = set(selected) - set(paths)
    if unknown:
        raise ValueError(f'Lessons not in manifest: {sorted(unknown)}')
    results = []
    for relative in selected:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != '.ipynb':
            raise ValueError(f'Invalid lesson path: {relative}')
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        started = time.monotonic()
        print(f'Running {relative}', flush=True)
        client = NotebookClient(
            notebook, timeout=180, kernel_name='python3',
            kernel_manager_class=LessonKernelManager,
            resources={'metadata': {'path': str(ROOT)}}, allow_errors=False,
        )
        executed = client.execute()
        code_cells = sum(cell.cell_type == 'code' and bool(cell.source.strip())
                         for cell in executed.cells)
        streams = [output.get('text', '') for cell in executed.cells if cell.cell_type == 'code'
                   for output in cell.get('outputs', []) if output.output_type == 'stream']
        result = {'path': relative, 'code_cells': code_cells,
                  'seconds': round(time.monotonic() - started, 2), 'text_output': ''.join(streams)}
        results.append(result)
        if output_dir:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(executed, destination)
        print(f'Passed: {code_cells} code cells in {result["seconds"]}s', flush=True)
    versions = {}
    for package in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient', 'torch']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    summary = {'python': sys.version.split()[0], 'versions': versions, 'lessons': results,
               'total_code_cells': sum(result['code_cells'] for result in results)}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Executed {len(results)} lessons / {summary["total_code_cells"]} code cells; sources unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'docs' / 'NEW_LESSONS.json')
    parser.add_argument('--only', nargs='+', help='Execute selected paths from the manifest')
    parser.add_argument('--report', type=Path, help='Optional JSON execution report')
    parser.add_argument('--output-dir', type=Path, help='Optional directory for executed notebook copies')
    options = parser.parse_args()
    execute(options.manifest, options.only, options.report, options.output_dir)

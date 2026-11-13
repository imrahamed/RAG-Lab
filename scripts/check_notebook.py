"""Validate the output-free notebook, execute offline, and refresh its HTML reading copy."""

import ast
import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/rag_walkthrough.ipynb"
PREVIEW = NOTEBOOK.with_suffix(".html")
FLAGS = {"RUN_LIVE", "INITIALIZE_REMOTE", "INGEST_SAMPLE", "RUN_EVALUATION"}


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for cell in code_cells:
        ast.parse(cell.source, filename=f"{NOTEBOOK.name}:{cell.id}")
        if cell.outputs or cell.execution_count is not None:
            raise RuntimeError("Clear notebook outputs before refreshing the offline preview.")

    controls = next(cell for cell in code_cells if cell.id == "controls")
    values = {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(controls.source).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in FLAGS
    }
    if set(values) != FLAGS or any(value is not False for value in values.values()):
        raise RuntimeError("All service-operation flags must be False in the shared notebook.")

    executed = copy.deepcopy(notebook)
    # Only the execution copy gets this guard. Native ZMQ kernel communication is unaffected.
    executed.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import socket\n"
            "from rag_lab.pipeline import RagService\n"
            "def _block_live_access(*args, **kwargs):\n"
            "    raise RuntimeError('Offline notebook validation blocks live service access')\n"
            "socket.create_connection = _block_live_access\n"
            "socket.socket.connect = _block_live_access\n"
            "socket.socket.connect_ex = _block_live_access\n"
            "RagService.__init__ = _block_live_access\n",
            id="offline-validation-guard",
        ),
    )
    with TemporaryDirectory(prefix="rag-lab-notebook-") as temporary:
        runtime = Path(temporary)
        kernel_dir = runtime / "kernels" / "rag-lab-validation"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "RAG Lab offline validation",
                    "language": "python",
                    "env": {
                        "IPYTHONDIR": str(runtime / "ipython"),
                        "JUPYTER_CONFIG_DIR": str(runtime / "config"),
                        "PYTHONUTF8": "1",
                    },
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="rag-lab-validation",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime / "kernels")]),
            connection_file=str(runtime / "connection.json"),
        )
        client = NotebookClient(
            executed, km=manager, timeout=90, resources={"metadata": {"path": str(ROOT)}}
        )
        try:
            client.execute()
        finally:
            # NBClient does not own an explicitly supplied KernelManager.
            if client.kc is not None:
                client.kc.stop_channels()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()

    executed.cells.pop(0)
    count = 0
    for cell in executed.cells:
        if cell.cell_type == "code":
            count += 1
            cell.execution_count = count
            for output in cell.outputs:
                if output.output_type == "execute_result":
                    output.execution_count = count
    exporter = HTMLExporter(template_name="lab", exclude_output_prompt=True)
    content, _ = exporter.from_notebook_node(executed)
    PREVIEW.write_text(content, encoding="utf-8")
    print(f"Validated notebook schema and {len(code_cells)} Python cells.")
    print("Executed all cells in offline mode with service/network access blocked.")
    print("Refreshed notebooks/rag_walkthrough.html; source notebook outputs remain clear.")


if __name__ == "__main__":
    main()

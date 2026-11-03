"""The deployment package must not depend on repository-only analysis tools."""

import ast
from pathlib import Path

import anomaly_detection


def test_runtime_does_not_import_research_or_plotting_tools():
    package = Path(anomaly_detection.__file__).parent
    research_dependencies = {"research", "matplotlib", "nbformat", "IPython"}
    for source in package.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [(node.module or "").split(".")[0]]
            else:
                continue
            assert research_dependencies.isdisjoint(imports), source

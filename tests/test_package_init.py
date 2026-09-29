import importlib


def test_package_version_present():
    pkg = importlib.import_module("src.backend_emralerts")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)

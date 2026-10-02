"""``from dakera import *`` must work whether or not optional integrations are installed."""

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_every_name_in_all_exists():
    import dakera

    missing = [name for name in dakera.__all__ if not hasattr(dakera, name)]
    assert missing == []


def test_star_import_without_optional_integration():
    # Simulate the TealTiger extra not being installed.
    result = _run(
        "import sys; sys.modules['dakera.integrations.tealtiger'] = None\n"
        "from dakera import *\n"
        "import dakera\n"
        "assert 'DakeraCostStorage' not in dakera.__all__\n"
        "assert 'DakeraClient' in dakera.__all__\n"
    )
    assert result.returncode == 0, result.stderr


def test_star_import_default_environment():
    result = _run("from dakera import *; DakeraClient")
    assert result.returncode == 0, result.stderr

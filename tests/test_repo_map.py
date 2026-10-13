"""The repository map (docs/repo-map.md) is generated from the code: it must be current, every file must say what it
is for, and every API path the browser calls must have a route that answers it."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def repo_map():
    spec = importlib.util.spec_from_file_location("repo_map", ROOT / "tools" / "repo_map.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


def test_repo_map_is_current():
    rm = repo_map()
    current = rm.OUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert current == rm.render(rm.scan()), "docs/repo-map.md is out of date: run  python tools/repo_map.py"


def test_every_file_says_what_it_is_for():
    missing = [p for p, f in repo_map().scan().files.items() if not f.purpose]
    assert not missing, ("give these files a header comment (TypeScript / CSS: the first comment after the imports;"
                         " Python: a module docstring): " + ", ".join(missing))


def test_every_api_call_has_a_route():
    unmatched = repo_map().scan().unmatched
    assert not unmatched, f"API paths called that no route serves: {unmatched}"

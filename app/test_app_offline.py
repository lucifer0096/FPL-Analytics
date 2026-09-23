"""Offline checks for the dashboard itself -- everything here runs WITHOUT
network access and WITHOUT the gitignored data/raw/ snapshot, i.e. under the
exact conditions a fresh CI checkout or a Streamlit Cloud deploy has (the
committed data/dashboard_* fallbacks only).

Complements app/test_shared_live.py (which needs FPL's real API and is
excluded from the default run by pytest.ini's `-m "not live"`): this file is
in the DEFAULT test run, because these are the checks that would have caught
a rendering bug before it reached production.

Run: pytest app/test_app_offline.py
"""

import ast
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared  # noqa: E402
from test_shared_live import _simulate_api_outage  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
HOME_PAGE = APP_DIR / "app.py"
HISTORICAL_PAGE = APP_DIR / "pages" / "1_Historical_and_Model.py"


def _shared_exports():
    """Every name shared.py currently binds -- defined in it, imported into
    it, or assigned at module level."""
    tree = ast.parse((APP_DIR / "shared.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_shared_provides_every_name_the_app_imports():
    """Import contract: shared.py is this app's re-export hub, and its own
    imports can look unused from inside it (they ARE unused inside it -- the
    pages consume them). Lint autofixes and drive-by cleanups both happily
    delete such a line, which only shows up as an ImportError the moment the
    app starts -- this pins that contract statically instead."""
    provided = _shared_exports()
    consumers = [HOME_PAGE, HISTORICAL_PAGE, APP_DIR / "test_shared_live.py", Path(__file__)]
    missing = []
    for consumer in consumers:
        tree = ast.parse(consumer.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "shared":
                for alias in node.names:
                    if alias.name != "*" and (alias.asname or alias.name) not in provided:
                        missing.append(f"{consumer.name}: {(alias.asname or alias.name)}")
    assert not missing, (
        "shared.py no longer provides name(s) other files import from it: "
        + ", ".join(sorted(missing))
    )


def _run(page: Path) -> AppTest:
    """Renders a page with every live FPL call forced to fail, so only the
    committed fallback files can satisfy it (the deploy/CI situation)."""
    originals = _simulate_api_outage()
    try:
        at = AppTest.from_file(str(page), default_timeout=300)
        at.run()
    finally:
        for name, fn in originals.items():
            setattr(shared.fpl_api, name, fn)
    assert not at.exception, "\n".join(e.message for e in at.exception)
    return at


def test_home_page_renders_offline():
    at = _run(HOME_PAGE)
    assert len(at.tabs) == 8, "expected all 8 Home tabs to render"


def test_transfers_tab_produces_a_result_offline():
    """Regression for the KeyError: 'sell_price' that reached production:
    the Transfers results block crashed on a renamed pandas column only when
    the optimizer actually returned a transfer, which no test ever drove."""
    originals = _simulate_api_outage()
    try:
        at = AppTest.from_file(str(HOME_PAGE), default_timeout=300)
        at.run()
        assert not at.exception, "\n".join(e.message for e in at.exception)

        # My Squad stores the real squad for Transfers to consume; if it
        # couldn't build one offline, that itself is worth knowing.
        assert "built_squad" in at.session_state, "My Squad did not store a squad for Transfers"
        assert at.session_state["built_squad_season_gw"] == "live_squad"

        # The default (gated) solve path.
        at.button(key="find_transfers_btn_home").click()
        at.run()
        assert not at.exception, "\n".join(e.message for e in at.exception)
        captions = [c.value for c in at.caption]
        results_shown = any(c.startswith("Bank before moves") for c in captions)
        holding = any("holding is optimal" in c for c in captions)
        assert results_shown or holding, (
            "Transfers rendered neither a result block nor a holding message; "
            f"captions were: {captions[-5:]}"
        )
        if results_shown:
            # The exact line that used to raise KeyError must render and be
            # a real formatted bank figure.
            bank_caption = next(c for c in captions if c.startswith("Bank before moves"))
            assert "£" in bank_caption and "Remaining bank" in bank_caption

        # Wildcard/Free Hit path (unlimited transfers) must not crash either.
        at.checkbox(key="unlimited_transfers_checkbox_home").set_value(True)
        at.run()
        at.button(key="find_transfers_btn_home").click()
        at.run()
        assert not at.exception, "\n".join(e.message for e in at.exception)
        captions = [c.value for c in at.caption]
        assert any("Unlimited transfers this gameweek" in c for c in captions), (
            "unlimited-transfers caption missing after enabling the checkbox"
        )
    finally:
        for name, fn in originals.items():
            setattr(shared.fpl_api, name, fn)


def test_historical_page_renders_offline():
    at = _run(HISTORICAL_PAGE)
    # _run already fails on any exception; also require the page's opening
    # section to have actually rendered rather than bailing out silently.
    assert [h.value for h in at.header if "Project overview" in h.value], (
        "Historical page did not render its Project overview section"
    )

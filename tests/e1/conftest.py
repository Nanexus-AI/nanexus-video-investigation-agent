"""E1 test gating: live Frigate API tests are opt-in only."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-frigate",
        action="store_true",
        default=False,
        help="Run authenticated live Frigate live API tests (requires private credentials)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live-frigate"):
        return
    # Deselect rather than skip/pass: default E0/E1 regression never depends on live Frigate host.
    items[:] = [item for item in items if "live_frigate" not in item.keywords]

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaderforge.render import RenderPool  # noqa: E402


@pytest.fixture(scope="session")
def pool():
    p = RenderPool(2, timeout=10.0)
    yield p
    p.close()

import os
import sys

# config.py lives at the repo root; ml_backend/ and data_pipeline/ are
# namespace packages (no __init__.py) whose own modules use a mix of flat
# ("from observability import ...") and package-style ("from ml_backend.x
# import ...") imports, so all three directories need to be importable.
_root = os.path.dirname(os.path.abspath(__file__))
for _p in (_root, os.path.join(_root, "ml_backend"), os.path.join(_root, "data_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

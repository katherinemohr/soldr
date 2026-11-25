# Where the cs349e-assn2 checkout is: $ASSN2_DIR if set, else the cs349e-assn2 submodule in this repo (if checked
# out), else a sibling checkout at ../cs349e-assn2.
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def assn2_dir():
    if os.environ.get("ASSN2_DIR"):
        return os.path.abspath(os.environ["ASSN2_DIR"])
    sub = os.path.join(_HERE, "cs349e-assn2")
    if os.path.exists(os.path.join(sub, "system", "engine.py")):
        return sub
    return os.path.normpath(os.path.join(_HERE, "..", "cs349e-assn2"))

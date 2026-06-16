from pathlib import Path

from setuptools_scm import get_version

try:
    __version__ = get_version(relative_to=Path(__file__).parent.parent)
except:
    __version__ = None

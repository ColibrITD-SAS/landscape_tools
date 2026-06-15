from pathlib import Path

from setuptools_scm import get_version

__version__ = get_version(relative_to=Path(__file__).parent.parent)

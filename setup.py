import sys

import setuptools
from setuptools import setup

__version__ = "0.1"

if sys.version_info < (3, 8):
    sys.exit("hypersync requires Python 3.8 or later.")

name = "hypersync"

version = __version__

authors = "Maxime Lucas"

author_email = "maxime.lucas.work@gmail.com"

url = "https://github.com/maximelucas/hypersync"

description = "HyperSync is a Python library for the simulation of synchronisation on complex systems with group (higher-order) interactions."


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
    return requires


extras_require = {
    dep: parse_requirements_file("requirements/" + dep + ".txt")
    for dep in ["developer", "documentation", "release", "test", "tutorial"]
}

install_requires = parse_requirements_file("requirements/default.txt")

license = "3-Clause BSD license"

setup(
    name=name,
    packages=setuptools.find_packages(),
    version=version,
    author=authors,
    author_email=author_email,
    url=url,
    description=description,
    install_requires=install_requires,
    extras_require=extras_require,
)
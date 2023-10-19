import pkg_resources

from . import analysis, drawing, simulation
from .analysis import *
from .drawing import *
from .simulation import *

__version__ = pkg_resources.require("hypersync")[0].version

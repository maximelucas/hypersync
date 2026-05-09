from importlib.metadata import version

from . import analysis, drawing, simulation
from .analysis import *
from .drawing import *
from .simulation import *

__version__ = version("hypersync")

import abc
from geomfum._registry import MeshPlotterRegistry, WhichRegistryMixins

'''
In this file i define the plotting logic used in geomfum
Since each plotting library has its own method of plotting, we define general functions that works with any library implemented 

'''

class ShapePlotter(abc.ABC):
    def plot(self, mesh):
        raise NotImplementedError("This method should be overridden by subclasses")

class MeshPlotter(WhichRegistryMixins, ShapePlotter):
    _Registry = MeshPlotterRegistry

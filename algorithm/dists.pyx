from cython_gsl cimport *
from libc.stdlib cimport calloc
from enum import Enum

cdef TDist * get_distribution(dist_t dist_id) nogil:
    cdef TDist * distribution = <TDist *> calloc(1, sizeof(TDist))
    if dist_id == UNIFORM:
        distribution.cdf = &uniform_cdf
        distribution.pdf = &uniform_pdf
    else:
        distribution.cdf = &uniform_cdf
        distribution.pdf = &uniform_pdf

    return distribution

class PyTDist(int, Enum):
    uniform = UNIFORM

class GenericDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        return None

    def pdf(self, x):
        return None

class Uniform(GenericDist):
    def cdf(self, x):
        return uniform_cdf(x, self.loc, self.scale)

    def pdf(self, x):
        return uniform_pdf(x, self.loc, self.scale)


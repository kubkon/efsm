from cython_gsl cimport *

cdef class GenericDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    cpdef double cdf(self, double x):
        return -1.0

    cpdef double pdf(self, double x):
        return -1.0

cdef class Uniform(GenericDist):
    cpdef double cdf(self, double x):
        if x < self.loc:
            return 0.0
        elif x > self.scale:
            return 1.0
        else:
            return (x - self.loc) / (self.scale - self.loc)

    cpdef double pdf(self, double x):
        if x < self.loc or x > self.scale:
            return 0.0
        else:
            return 1 / (self.scale - self.loc)


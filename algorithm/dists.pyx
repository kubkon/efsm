from cython_gsl cimport *

cdef double uniform_cdf(double x, double a, double b) nogil:
    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return (x - a) / (b - a)

cdef double uniform_pdf(double x, double a, double b) nogil:
    if x < a or x > b:
        return 0.0
    else:
        return 1 / (b - a)

class Uniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def cdf(self, x):
        return uniform_cdf(x, self.a, self.b)

    def pdf(self, x):
        return uniform_pdf(x, self.a, self.b)


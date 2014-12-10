from cython_gsl cimport *
from libc.stdlib cimport calloc
from enum import IntEnum

cdef TDist * get_distribution(supported_dists dist_id) nogil:
    cdef TDist * distribution = <TDist *> calloc(1, sizeof(TDist))
    if dist_id == UNIFORM:
        distribution.cdf = &uniform_cdf
        distribution.pdf = &uniform_pdf
    elif dist_id == NORMAL:
        distribution.cdf = &truncated_normal_cdf
        distribution.pdf = &truncated_normal_pdf
    else:
        distribution.cdf = &uniform_cdf
        distribution.pdf = &uniform_pdf

    return distribution

cdef double uniform_cdf(double x, TDistParams * params) nogil:
    cdef:
        double a = params.a
        double b = params.b

    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return (x - a) / (b - a)

cdef double uniform_pdf(double x, TDistParams * params) nogil:
    cdef:
        double a = params.a
        double b = params.b

    if x < a or x > b:
        return 0.0
    else:
        return 1 / (b - a)

cdef double truncated_normal_cdf(double x, TDistParams * params) nogil:
    cdef:
        double loc = params.loc
        double scale = params.scale
        double a = params.a
        double b = params.b
        double epsilon = (x - loc) / scale
        double alpha = (a - loc) / scale
        double beta = (b - loc) / scale

    return ((standard_normal_cdf(epsilon) - standard_normal_cdf(alpha))
           /(standard_normal_cdf(beta) - standard_normal_cdf(alpha)))

cdef double truncated_normal_pdf(double x, TDistParams * params) nogil:
    cdef:
        double loc = params.loc
        double scale = params.scale
        double a = params.a
        double b = params.b
        double epsilon = (x - loc) / scale
        double alpha = (a - loc) / scale
        double beta = (b - loc) / scale

    return (standard_normal_pdf(epsilon)
           /(scale * (standard_normal_cdf(beta) - standard_normal_cdf(alpha))))

class SupportedDistributions(IntEnum):
    uniform = UNIFORM
    normal = NORMAL

def py_get_distribution(dist_id):
    dist = None
    if dist_id == SupportedDistributions.uniform:
        dist = Uniform
    elif dist_id == SupportedDistributions.normal:
        dist = TruncatedNormal
    else:
        dist = Uniform
    return dist

cdef class GenericDist:
    cdef TDistParams params

    def __init__(self, loc, scale, a, b):
        self.params.loc = loc
        self.params.scale = scale
        self.params.a = a
        self.params.b = b

    def cdf(self, x):
        return None

    def pdf(self, x):
        return None

cdef class Uniform(GenericDist):
    def cdf(self, x):
        return uniform_cdf(x, &self.params)

    def pdf(self, x):
        return uniform_pdf(x, &self.params)

cdef class TruncatedNormal(GenericDist):
    def cdf(self, x):
        return truncated_normal_cdf(x, &self.params)

    def pdf(self, x):
        return truncated_normal_pdf(x, &self.params)


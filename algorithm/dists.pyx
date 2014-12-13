from cython_gsl cimport *
from libc.stdlib cimport calloc

cdef double uniform_cdf(double x, DistParams * params) nogil:
    cdef:
        double a = params.a
        double b = params.b

    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return (x - a) / (b - a)

cdef double uniform_pdf(double x, DistParams * params) nogil:
    cdef:
        double a = params.a
        double b = params.b

    if x < a or x > b:
        return 0.0
    else:
        return 1 / (b - a)

cdef double truncated_normal_cdf(double x, DistParams * params) nogil:
    cdef:
        double loc = params.loc
        double scale = params.scale
        double a = params.a
        double b = params.b
        double epsilon = (x - loc) / scale
        double alpha = (a - loc) / scale
        double beta = (b - loc) / scale

    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return ((standard_normal_cdf(epsilon) - standard_normal_cdf(alpha))
               /(standard_normal_cdf(beta) - standard_normal_cdf(alpha)))

cdef double truncated_normal_pdf(double x, DistParams * params) nogil:
    cdef:
        double loc = params.loc
        double scale = params.scale
        double a = params.a
        double b = params.b
        double epsilon = (x - loc) / scale
        double alpha = (a - loc) / scale
        double beta = (b - loc) / scale

    if x < a or x > b:
        return 0.0
    else:
        return (standard_normal_pdf(epsilon)
               /(scale * (standard_normal_cdf(beta) - standard_normal_cdf(alpha))))

cdef Dist get_distribution(CSupportedDists id) nogil:
    cdef Dist dist
    dist.cdf = &uniform_cdf
    dist.pdf = &uniform_pdf

    if id == TRUNC_NORMAL:
        dist.cdf = &truncated_normal_cdf
        dist.pdf = &truncated_normal_pdf
    else:
        pass

    return dist

cdef class GenericDist:
    cdef DistParams params

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
    c_id = UNIFORM

    def cdf(self, x):
        return uniform_cdf(x, &self.params)

    def pdf(self, x):
        return uniform_pdf(x, &self.params)

cdef class TruncatedNormal(GenericDist):
    c_id = TRUNC_NORMAL

    def cdf(self, x):
        return truncated_normal_cdf(x, &self.params)

    def pdf(self, x):
        return truncated_normal_pdf(x, &self.params)


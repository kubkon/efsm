from libc.math cimport exp, sqrt, pow, erf

ctypedef struct TDistParams:
    double loc
    double scale
    double a
    double b

ctypedef struct TDist:
    double cdf(double, TDistParams*) nogil
    double pdf(double, TDistParams*) nogil

ctypedef enum supported_dists:
    UNIFORM      = 0
    TRUNC_NORMAL = 1

cdef TDist * get_distribution(supported_dists dist_id) nogil

cdef double uniform_cdf(double x, TDistParams * params) nogil
cdef double uniform_pdf(double x, TDistParams * params) nogil

cdef inline double standard_normal_cdf(double x) nogil:
    return 0.5 * (1 + erf(x / sqrt(2)))

cdef inline double standard_normal_pdf(double x) nogil:
    cdef double pi = 3.14159265
    return exp(-0.5 * pow(x, 2)) / sqrt(2 * pi)

cdef double truncated_normal_cdf(double x, TDistParams * params) nogil
cdef double truncated_normal_pdf(double x, TDistParams * params) nogil


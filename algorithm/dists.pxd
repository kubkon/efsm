from libc.math cimport exp, sqrt, pow, erf

ctypedef struct DistParams:
    double loc
    double scale
    double a
    double b

ctypedef struct Dist:
    double cdf(double, DistParams *) nogil
    double pdf(double, DistParams *) nogil

ctypedef struct BiddingParams:
    DistParams dist_params
    Dist dist

cdef double uniform_cdf(double x, DistParams * params) nogil
cdef double uniform_pdf(double x, DistParams * params) nogil

cdef inline double standard_normal_cdf(double x) nogil:
    return 0.5 * (1 + erf(x / sqrt(2)))

cdef inline double standard_normal_pdf(double x) nogil:
    cdef double pi = 3.14159265
    return exp(-0.5 * pow(x, 2)) / sqrt(2 * pi)

cdef double truncated_normal_cdf(double x, DistParams * params) nogil
cdef double truncated_normal_pdf(double x, DistParams * params) nogil

ctypedef enum CSupportedDists:
    UNIFORM = 0
    TRUNC_NORMAL = 1

cdef Dist get_distribution(CSupportedDists) nogil


ctypedef struct TDist:
    double cdf(double, double, double) nogil
    double pdf(double, double, double) nogil

ctypedef enum dist_t:
    UNIFORM = 0

cdef TDist * get_distribution(dist_t dist_id) nogil

cdef inline double uniform_cdf(double x, double a, double b) nogil:
    if x < a:
        return 0.0
    elif x > b:
        return 1.0
    else:
        return (x - a) / (b - a)

cdef inline double uniform_pdf(double x, double a, double b) nogil:
    if x < a or x > b:
        return 0.0
    else:
        return 1 / (b - a)


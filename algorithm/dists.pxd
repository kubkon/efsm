cdef class GenericDist:
    cdef double loc
    cdef double scale

    cpdef double cdf(self, double x)

    cpdef double pdf(self, double x)

cdef class Uniform(GenericDist):
    pass


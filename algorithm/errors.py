class Error(Exception):
    message = None

    def __str__(self):
        return repr(self.message)

class GSLZeroDivision(Error):
    message = "GSL encountered zero division"

class GSLFailure(Error):
    message = "GSL failed"

class GSLMemoryFailure(Error):
    message = "GSL failed to allocate necessary memory"

class EFSMTruncationIndexExceeded(Error):
    message = "Index of truncation exceeds permissible range"

class EFSMCompilationError(Error):
    message = "Import error, perhaps you forgot to run 'make'?"

class EFSMConvergenceParam(Error):
    message = "Could not estimate convergence parameter"

class EFSMMaximumIterationExceeded(Error):
    message = "Exceeded maximum number of iterations allowed"

class EFSMFailure(Error):
    message = "EFSM failed"

gsl_error_mapping = {
    -1: GSLFailure,
    8: GSLMemoryFailure,
    12: GSLZeroDivision
}


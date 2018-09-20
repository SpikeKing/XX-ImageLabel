# Level 0
class BaseError(Exception):
    def __init__(self, errno, msg=''):
        self.errno = errno
        self.msg = msg
    def __str__(self):
        _s = self.__class__.__name__ + ': errno=' + repr(self.errno)
        if self.msg != '':
            _s += ', msg="%s"' % self.msg
        return _s
# Level 1
#  Python has IOError already
class DataIOError(BaseError):
    TIMEOUT = 0
    REQUEST_FAILED = 1
    SERVER_NOT_AVAILABLE = 2
    FILE_NOT_FOUND = 3
    pass
class DataFormatError(BaseError):
    SHAPE_MISMATCH = 0
    DTYPE_MISMATCH = 1
    DECODE_FAILED = 2
    CHANNEL_MISMATCH = 3
    RESIZE_MISMATCH = 4
    pass
class GPUError(BaseError):
    MEMORY_OUT = 0
    DEVICE_NOT_FOUND = 1
    pass
class UnknownError(BaseError):
    pass

"""
Backboard API exceptions
"""

class BackboardError(Exception):
    """Base exception for all Backboard API errors"""
    pass


class BackboardAPIError(BackboardError):
    """Raised when the API returns an error response"""
    
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class BackboardValidationError(BackboardAPIError):
    """Raised when request validation fails (400 status code)"""
    pass


class BackboardNotFoundError(BackboardAPIError):
    """Raised when a resource is not found (404 status code)"""
    pass


class BackboardRateLimitError(BackboardAPIError):
    """Raised when rate limit is exceeded (429 status code)"""
    pass


class BackboardServerError(BackboardAPIError):
    """Raised when server returns 5xx error"""
    pass

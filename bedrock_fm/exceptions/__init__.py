__all__ = [
    "BedrockExtraArgsError",
    "BedrockInvalidModelError"
]

class BedrockExtraArgsError(Exception):
    def __init__(self, message):
        super().__init__(message)

class BedrockInvalidModelError(Exception):
    def __init__(self, message):
        super().__init__(message)

class BedrockArgsError(Exception):
    def __init__(self, message):
        super().__init__(message)
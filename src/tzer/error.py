# Also, there's a timeout error which is managed by subprocess module.

class IncorrectResult(Exception):
    pass

class PerfDegradation(Exception):
    pass

class RuntimeFailure(Exception):
    pass

# Timeout...
class MaybeDeadLoop(Exception):
    pass

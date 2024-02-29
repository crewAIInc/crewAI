class Logger:
    def __init__(self, verbose_level=0):
        # If verbose_level is a boolean and is True, set it to 2, otherwise keep its original value
        verbose_level = (
            2 if isinstance(verbose_level, bool) and verbose_level else verbose_level
        )
        # Store the verbose_level in an instance variable
        self.verbose_level = verbose_level

    def log(self, level, message):
        # Define a mapping from log levels to their corresponding numeric values
        level_map = {"debug": 1, "info": 2}
        
        # If verbose_level is set and the numeric value of the provided level is less than or equal to verbose_level,
        # print the log message. If the level is not in the level_map, default to 0.
        if self.verbose_level and level_map.get(level, 0) <= self.verbose_level:
            print(f"[{level.upper()}]: {message}")

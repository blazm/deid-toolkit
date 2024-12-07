import os
from datetime import datetime
class DeidtoolkitError(Exception):
    """Base class for pipeline-related errors."""
    def __init__(self, message, module=None, details=None):
        super().__init__(message)
        self.module = module  
        self.details = details
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __str__(self):
        base_message = "Deidtoolkit Error"
        if self.module:
            base_message += f" in module {self.module}: "
        base_message += super().__str__()
        if self.details:
            base_message += f" | Details: {self.details}"
        return base_message
    
    def to_dict(self):
        # Convert the error details to a dictionary
        return {
            "module": self.module if self.module else "N/A",
            "message": str(self),
            "timestamp": self.timestamp,
            "details": self.details if self.details else "N/A"
        }


        


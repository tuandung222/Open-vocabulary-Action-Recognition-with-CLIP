"""
Configuration utilities.
"""

class Config:
    """
    Simple configuration class.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self):
        """String representation of the config."""
        attrs = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f"Config({attrs})"
        
    def get(self, key, default=None):
        """Get a configuration value with a default."""
        return getattr(self, key, default)
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
        
    def update(self, **kwargs):
        """Update config with new values."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
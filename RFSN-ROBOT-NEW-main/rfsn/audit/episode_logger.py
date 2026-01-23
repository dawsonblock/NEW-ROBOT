"""
Deterministic Episode Logger
============================
Logs every step for reproducibility.
"""

import json
from typing import Any, Dict


class EpisodeLogger:
    """
    Deterministic logger for episode replay.
    
    Log every step:
    - contacts
    - envelope violations
    - reward components
    - state transitions
    """
    
    def __init__(self, path: str):
        """
        Args:
            path: Output file path (JSONL format)
        """
        self.path = path
        self.f = open(path, "w")

    def log(self, step: int, data: Dict[str, Any]):
        """
        Log a single step.
        
        Args:
            step: Step number
            data: Step data dictionary
        """
        data["step"] = step
        self.f.write(json.dumps(data) + "\n")

    def close(self):
        """Close the log file."""
        self.f.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

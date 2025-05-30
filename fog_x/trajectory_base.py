from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text, Union

import numpy as np


class TrajectoryInterface(ABC):
    """
    Abstract base class defining the interface for trajectory objects.
    This allows for better testing and dependency injection.
    """

    @abstractmethod
    def add(self,
            feature: str,
            data: Any,
            timestamp: Optional[int] = None) -> None:
        """Add a single feature value to the trajectory."""
        pass

    @abstractmethod
    def add_by_dict(self,
                    data: Dict[str, Any],
                    timestamp: Optional[int] = None) -> None:
        """Add multiple features from a dictionary to the trajectory."""
        pass

    @abstractmethod
    def load(self,
             save_to_cache: bool = True,
             return_type: str = "numpy") -> Union[Dict, Any]:
        """Load the trajectory data."""
        pass

    @abstractmethod
    def close(self, compact: bool = True) -> None:
        """Close the trajectory file."""
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Get a feature from the trajectory."""
        pass


class FileSystemInterface(ABC):
    """Abstract interface for file system operations to enable testing with mocks."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create directories."""
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        """Remove a file."""
        pass

    @abstractmethod
    def rename(self, src: str, dst: str) -> None:
        """Rename a file."""
        pass


class DefaultFileSystem(FileSystemInterface):
    """Default implementation using standard file system operations."""

    def exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        import os

        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path: str) -> None:
        import os

        os.remove(path)

    def rename(self, src: str, dst: str) -> None:
        import os

        os.rename(src, dst)


class TimeProvider(ABC):
    """Abstract interface for time operations to enable testing."""

    @abstractmethod
    def time(self) -> float:
        """Get current time."""
        pass


class DefaultTimeProvider(TimeProvider):
    """Default implementation using standard time operations."""

    def time(self) -> float:
        import time

        return time.time()

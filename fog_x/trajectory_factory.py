"""Factory for creating trajectory instances with dependency injection."""

from typing import Optional, Text
from .trajectory_base import TrajectoryInterface, FileSystemInterface, TimeProvider, DefaultFileSystem, DefaultTimeProvider


class TrajectoryFactory:
    """Factory for creating trajectory instances with configurable dependencies."""
    
    def __init__(
        self,
        filesystem: Optional[FileSystemInterface] = None,
        time_provider: Optional[TimeProvider] = None
    ):
        self.filesystem = filesystem or DefaultFileSystem()
        self.time_provider = time_provider or DefaultTimeProvider()
    
    def create_trajectory(
        self,
        path: Text,
        mode: str = "r",
        cache_dir: Optional[Text] = "/tmp/fog_x/cache/",
        lossy_compression: bool = True,
        feature_name_separator: Text = "/",
    ) -> TrajectoryInterface:
        """Create a trajectory instance with injected dependencies."""
        from .trajectory import Trajectory
        
        # Create trajectory with dependency injection
        trajectory = Trajectory(
            path=path,
            mode=mode,
            cache_dir=cache_dir,
            lossy_compression=lossy_compression,
            feature_name_separator=feature_name_separator,
            filesystem=self.filesystem,
            time_provider=self.time_provider,
        )
        
        return trajectory


# Global factory instance for backwards compatibility
default_factory = TrajectoryFactory()


def create_trajectory(
    path: Text,
    mode: str = "r",
    cache_dir: Optional[Text] = "/tmp/fog_x/cache/",
    lossy_compression: bool = True,
    feature_name_separator: Text = "/",
) -> TrajectoryInterface:
    """Convenience function to create trajectory with default dependencies."""
    return default_factory.create_trajectory(
        path=path,
        mode=mode,
        cache_dir=cache_dir,
        lossy_compression=lossy_compression,
        feature_name_separator=feature_name_separator,
    ) 
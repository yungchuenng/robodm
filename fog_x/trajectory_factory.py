"""Factory for creating trajectory instances with dependency injection."""

from typing import Any, Dict, Optional, Text
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
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        feature_name_separator: Text = "/",
    ) -> TrajectoryInterface:
        """
        Create a trajectory instance with injected dependencies.
        
        Args:
            path (Text): Path to trajectory file
            mode (str): File mode ("r" or "w")
            video_codec (str): Video codec to use ("auto", "rawvideo", "h264", "h265", "libaom-av1", "ffv1")
            codec_options (Dict[str, Any]): Additional codec-specific options
            feature_name_separator (Text): Delimiter for feature names
        """
        from .trajectory import Trajectory
        
        # Create trajectory with dependency injection
        trajectory = Trajectory(
            path=path,
            mode=mode,
            video_codec=video_codec,
            codec_options=codec_options,
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
    video_codec: str = "auto",
    codec_options: Optional[Dict[str, Any]] = None,
    feature_name_separator: Text = "/",
) -> TrajectoryInterface:
    """
    Convenience function to create trajectory with default dependencies.
    
    Args:
        path (Text): Path to trajectory file
        mode (str): File mode ("r" or "w")
        video_codec (str): Video codec to use ("auto", "rawvideo", "h264", "h265", "libaom-av1", "ffv1")
        codec_options (Dict[str, Any]): Additional codec-specific options
        feature_name_separator (Text): Delimiter for feature names
    """
    return default_factory.create_trajectory(
        path=path,
        mode=mode,
        video_codec=video_codec,
        codec_options=codec_options,
        feature_name_separator=feature_name_separator,
    ) 
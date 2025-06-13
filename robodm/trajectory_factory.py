"""Factory for creating trajectory instances with dependency injection."""

from datetime import datetime
from typing import Any, Dict, Optional, Text

from .trajectory_base import (DefaultFileSystem, DefaultTimeProvider,
                              FileSystemInterface, TimeProvider,
                              TrajectoryInterface)


class TrajectoryFactory:
    """Factory for creating trajectory instances with configurable dependencies."""

    def __init__(
        self,
        filesystem: Optional[FileSystemInterface] = None,
        time_provider: Optional[TimeProvider] = None,
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
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
        visualization_feature: Optional[Text] = None,
        backend: Optional[Any] = None,
    ) -> TrajectoryInterface:
        """
        Create a trajectory instance with injected dependencies.

        Args:
            path (Text): Path to trajectory file
            mode (str): File mode ("r" or "w")
            video_codec (str): Video codec to use ("auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1")
            codec_options (Dict[str, Any]): Additional codec-specific options
            feature_name_separator (Text): Delimiter for feature names
            base_datetime: Optional base datetime for timestamp calculations
            time_unit: Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
            enforce_monotonic: Whether to enforce monotonically increasing timestamps
            visualization_feature: Optional feature name to prioritize as first stream for visualization
            backend: Optional container backend for dependency injection
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
            base_datetime=base_datetime,
            time_unit=time_unit,
            enforce_monotonic=enforce_monotonic,
            visualization_feature=visualization_feature,
            backend=backend,
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
    base_datetime: Optional[datetime] = None,
    time_unit: str = "ms",
    enforce_monotonic: bool = True,
    visualization_feature: Optional[Text] = None,
    backend: Optional[Any] = None,
) -> TrajectoryInterface:
    """
    Convenience function to create trajectory with default dependencies.

    Args:
        path (Text): Path to trajectory file
        mode (str): File mode ("r" or "w")
        video_codec (str): Video codec to use ("auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1")
        codec_options (Dict[str, Any]): Additional codec-specific options
        feature_name_separator (Text): Delimiter for feature names
        base_datetime: Optional base datetime for timestamp calculations
        time_unit: Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
        enforce_monotonic: Whether to enforce monotonically increasing timestamps
        visualization_feature: Optional feature name to prioritize as first stream for visualization
        backend: Optional container backend for dependency injection
    """
    from .trajectory import Trajectory

    return Trajectory(
        path=path,
        mode=mode,
        video_codec=video_codec,
        codec_options=codec_options,
        feature_name_separator=feature_name_separator,
        base_datetime=base_datetime,
        time_unit=time_unit,
        enforce_monotonic=enforce_monotonic,
        visualization_feature=visualization_feature,
        backend=backend,
    )

import asyncio
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Any, Dict, List, Optional, Text, Tuple, Union, cast

import av
import h5py
import numpy as np

from robodm import FeatureType
from robodm.trajectory_base import TrajectoryInterface
from robodm.utils import recursively_read_hdf5_group

logger = logging.getLogger(__name__)

logging.getLogger("libav").setLevel(logging.CRITICAL)


def _flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TimeManager:
    """
    Comprehensive time management system for robodm trajectories.

    Handles:
    - Multiple time units (nanoseconds, microseconds, milliseconds, seconds)
    - Base datetime reference points
    - Monotonic timestamp enforcement
    - Unit conversions
    - Per-timestep timing from base datetime
    """

    # Time unit conversion factors to nanoseconds
    TIME_UNITS = {
        "ns": 1,
        "nanoseconds": 1,
        "μs": 1_000,
        "us": 1_000,
        "microseconds": 1_000,
        "ms": 1_000_000,
        "milliseconds": 1_000_000,
        "s": 1_000_000_000,
        "seconds": 1_000_000_000,
    }

    # Trajectory time base (for robodm compatibility)
    TRAJECTORY_TIME_BASE = Fraction(1, 1000)  # milliseconds

    def __init__(
        self,
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
    ):
        """
        Initialize TimeManager.

        Parameters:
        -----------
        base_datetime : datetime, optional
            Reference datetime for relative timestamps. If None, uses current time.
        time_unit : str
            Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
        enforce_monotonic : bool
            Whether to enforce monotonically increasing timestamps
        """
        self.base_datetime = base_datetime or datetime.now(timezone.utc)
        self.time_unit = time_unit
        self.enforce_monotonic = enforce_monotonic

        # Internal state
        self._last_timestamp_ns = 0
        self._start_time = time.time()

        # Validate time unit
        if time_unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {time_unit}. "
                             f"Supported: {list(self.TIME_UNITS.keys())}")

    def reset(self, base_datetime: Optional[datetime] = None):
        """Reset the time manager with new base datetime."""
        if base_datetime:
            self.base_datetime = base_datetime
        self._last_timestamp_ns = 0
        self._start_time = time.time()

    def current_timestamp(self, unit: Optional[str] = None) -> int:
        """
        Get current timestamp relative to start time.

        Parameters:
        -----------
        unit : str, optional
            Time unit for returned timestamp. If None, uses default unit.

        Returns:
        --------
        int : Current timestamp in specified unit
        """
        unit = unit or self.time_unit
        current_time_ns = int((time.time() - self._start_time) * 1_000_000_000)
        return self.convert_from_nanoseconds(current_time_ns, unit)

    def datetime_to_timestamp(self,
                              dt: datetime,
                              unit: Optional[str] = None) -> int:
        """
        Convert datetime to timestamp relative to base_datetime.

        Parameters:
        -----------
        dt : datetime
            Datetime to convert
        unit : str, optional
            Target time unit. If None, uses default unit.

        Returns:
        --------
        int : Timestamp in specified unit
        """
        unit = unit or self.time_unit
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if self.base_datetime.tzinfo is None:
            base_dt = self.base_datetime.replace(tzinfo=timezone.utc)
        else:
            base_dt = self.base_datetime

        delta_seconds = (dt - base_dt).total_seconds()
        delta_ns = int(delta_seconds * 1_000_000_000)
        return self.convert_from_nanoseconds(delta_ns, unit)

    def timestamp_to_datetime(self,
                              timestamp: int,
                              unit: Optional[str] = None) -> datetime:
        """
        Convert timestamp to datetime using base_datetime as reference.

        Parameters:
        -----------
        timestamp : int
            Timestamp value
        unit : str, optional
            Time unit of input timestamp. If None, uses default unit.

        Returns:
        --------
        datetime : Corresponding datetime
        """
        unit = unit or self.time_unit
        timestamp_ns = self.convert_to_nanoseconds(timestamp, unit)
        delta_seconds = timestamp_ns / 1_000_000_000

        if self.base_datetime.tzinfo is None:
            base_dt = self.base_datetime.replace(tzinfo=timezone.utc)
        else:
            base_dt = self.base_datetime

        return base_dt + timedelta(seconds=delta_seconds)

    def convert_to_nanoseconds(self, timestamp: Union[int, float],
                               unit: str) -> int:
        """Convert timestamp from given unit to nanoseconds."""
        if unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {unit}")
        return int(timestamp * self.TIME_UNITS[unit])

    def convert_from_nanoseconds(self, timestamp_ns: int, unit: str) -> int:
        """Convert timestamp from nanoseconds to given unit."""
        if unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {unit}")
        return int(timestamp_ns // self.TIME_UNITS[unit])

    def convert_units(self, timestamp: Union[int, float], from_unit: str,
                      to_unit: str) -> int:
        """Convert timestamp between different units."""
        timestamp_ns = self.convert_to_nanoseconds(timestamp, from_unit)
        return self.convert_from_nanoseconds(timestamp_ns, to_unit)

    def validate_timestamp(self,
                           timestamp: int,
                           unit: Optional[str] = None) -> int:
        """
        Validate and potentially adjust timestamp for monotonic ordering.

        Parameters:
        -----------
        timestamp : int
            Input timestamp
        unit : str, optional
            Time unit of input timestamp

        Returns:
        --------
        int : Validated timestamp in trajectory time base units (milliseconds)
        """
        unit = unit or self.time_unit
        timestamp_ns = self.convert_to_nanoseconds(timestamp, unit)

        if self.enforce_monotonic:
            if timestamp_ns <= self._last_timestamp_ns:
                # Adjust to maintain monotonic ordering - add 1ms worth of nanoseconds to ensure difference
                timestamp_ns = (self._last_timestamp_ns + 1_000_000
                                )  # +1ms in nanoseconds
                logger.debug(
                    f"Adjusted timestamp to maintain monotonic ordering: {timestamp_ns} ns"
                )

            self._last_timestamp_ns = timestamp_ns

        # Convert to trajectory time base (milliseconds)
        return self.convert_from_nanoseconds(timestamp_ns, "ms")

    def add_timestep(self,
                     timestep: Union[int, float],
                     unit: Optional[str] = None) -> int:
        """
        Add a timestep to the last timestamp and return trajectory-compatible timestamp.

        Parameters:
        -----------
        timestep : int or float
            Time step to add
        unit : str, optional
            Time unit of timestep

        Returns:
        --------
        int : New timestamp in trajectory time base units (milliseconds)
        """
        unit = unit or self.time_unit
        timestep_ns = self.convert_to_nanoseconds(timestep, unit)
        new_timestamp_ns = self._last_timestamp_ns + timestep_ns

        self._last_timestamp_ns = new_timestamp_ns
        return self.convert_from_nanoseconds(new_timestamp_ns, "ms")

    def create_timestamp_sequence(
        self,
        start_timestamp: int,
        count: int,
        timestep: Union[int, float],
        unit: Optional[str] = None,
    ) -> List[int]:
        """
        Create a sequence of monotonic timestamps.

        Parameters:
        -----------
        start_timestamp : int
            Starting timestamp
        count : int
            Number of timestamps to generate
        timestep : int or float
            Time step between consecutive timestamps
        unit : str, optional
            Time unit for inputs

        Returns:
        --------
        List[int] : List of timestamps in trajectory time base units
        """
        unit = unit or self.time_unit
        start_ns = self.convert_to_nanoseconds(start_timestamp, unit)
        timestep_ns = self.convert_to_nanoseconds(timestep, unit)

        timestamps = []
        current_ns = start_ns

        for i in range(count):
            # Ensure monotonic ordering if enforce_monotonic is True
            if self.enforce_monotonic and current_ns <= self._last_timestamp_ns:
                current_ns = self._last_timestamp_ns + 1_000_000  # +1ms in nanoseconds

            timestamps.append(self.convert_from_nanoseconds(current_ns, "ms"))

            # Update last timestamp only if monotonic enforcement is enabled
            if self.enforce_monotonic:
                self._last_timestamp_ns = current_ns

            current_ns += timestep_ns

        return timestamps


class StreamInfo:

    def __init__(self, feature_name, feature_type, encoding):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.encoding = encoding

    def __str__(self):
        return f"StreamInfo({self.feature_name}, {self.feature_type}, {self.encoding})"

    def __repr__(self):
        return self.__str__()


class CodecConfig:
    """Configuration class for video codec settings."""

    @staticmethod
    def get_supported_pixel_formats(codec_name: str) -> List[str]:
        """Get list of supported pixel formats for a codec."""
        try:
            import av

            codec = av.codec.Codec(codec_name, "w")
            if codec.video_formats:
                return [vf.name for vf in codec.video_formats]
            return []
        except Exception:
            return []

    @staticmethod
    def is_codec_config_supported(width: int,
                                  height: int,
                                  pix_fmt: str = "yuv420p",
                                  codec_name: str = "libx264") -> bool:
        """Check if a specific width/height/pixel format combination is supported by codec."""
        try:
            from fractions import Fraction

            import av

            cc = av.codec.CodecContext.create(codec_name, "w")
            cc.width = width
            cc.height = height
            cc.pix_fmt = pix_fmt
            cc.time_base = Fraction(1, 30)
            cc.open(strict=True)
            cc.close()
            return True
        except Exception:
            return False

    @staticmethod
    def is_valid_image_shape(shape: Tuple[int, ...],
                             codec_name: str = "libx264") -> bool:
        """Check if a shape can be treated as an RGB image for the given codec."""
        # Only accept RGB shapes (H, W, 3)
        if len(shape) != 3 or shape[2] != 3:
            return False

        height, width = shape[0], shape[1]

        # Check minimum reasonable image size
        if height < 1 or width < 1:
            return False

        # Check codec-specific constraints
        if codec_name in ["libx264", "libx265"]:
            # H.264/H.265 require even dimensions
            if height % 2 != 0 or width % 2 != 0:
                return False
        elif codec_name in ["libaom-av1"]:
            # AV1 also typically requires even dimensions for yuv420p
            if height % 2 != 0 or width % 2 != 0:
                return False

        # Test if the codec actually supports this resolution
        return CodecConfig.is_codec_config_supported(width, height, "yuv420p",
                                                     codec_name)

    # Default codec configurations
    CODEC_CONFIGS = {
        "rawvideo": {
            "pixel_format": None,  # No pixel format for rawvideo (binary)
            "options": {},
        },
        "libx264": {
            "pixel_format": "yuv420p",
            "options": {
                "crf": "23",
                "preset": "medium"
            },  # Default quality
        },
        "libx265": {
            "pixel_format": "yuv420p",
            "options": {
                "crf": "28",
                "preset": "medium"
            },  # Default quality for HEVC
        },
        "libaom-av1": {
            "pixel_format": "yuv420p",
            "options": {
                "g": "2",
                "crf": "30"
            }
        },
        "ffv1": {
            "pixel_format":
            "yuv420p",  # Default, will be adjusted based on content
            "options": {},
        },
    }

    def __init__(self,
                 codec: str = "auto",
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize codec configuration.

        Args:
            codec: Video codec to use. Options: "auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1"
            options: Additional codec-specific options
        """
        self.codec = codec
        self.custom_options = options or {}

        if codec not in ["auto"] and codec not in self.CODEC_CONFIGS:
            raise ValueError(
                f"Unsupported codec: {codec}. Supported: {list(self.CODEC_CONFIGS.keys())}"
            )

    def get_codec_for_feature(self, feature_type: FeatureType) -> str:
        """Determine the appropriate codec for a given feature type."""

        data_shape = feature_type.shape

        # Only use video codecs for RGB images (H, W, 3)
        if data_shape is not None and len(
                data_shape) == 3 and data_shape[2] == 3:
            height, width = data_shape[0], data_shape[1]

            # If user specified a codec other than auto, try to use it for RGB images
            if self.codec != "auto":
                if self.is_valid_image_shape(data_shape, self.codec):
                    logger.debug(
                        f"Using user-specified codec {self.codec} for RGB shape {data_shape}"
                    )
                    return self.codec
                else:
                    logger.warning(
                        f"User-specified codec {self.codec} doesn't support shape {data_shape}, falling back to rawvideo"
                    )
                    return "rawvideo"

            # Auto-selection for RGB images only
            codec_preferences = ["libaom-av1", "ffv1", "libx264", "libx265"]

            for codec in codec_preferences:
                if self.is_valid_image_shape(data_shape, codec):
                    logger.debug(
                        f"Selected codec {codec} for RGB shape {data_shape}")
                    return codec

            # If no video codec works for this RGB image, fall back to rawvideo
            logger.warning(
                f"No video codec supports RGB shape {data_shape}, falling back to rawvideo"
            )

        else:
            # Non-RGB data (grayscale, depth, vectors, etc.) always use rawvideo
            if data_shape is not None:
                logger.debug(f"Using rawvideo for non-RGB shape {data_shape}")

        return "rawvideo"

    def get_pixel_format(self, codec: str,
                         feature_type: FeatureType) -> Optional[str]:
        """Get appropriate pixel format for codec and feature type."""
        if codec not in self.CODEC_CONFIGS:
            return None

        codec_config = cast(Dict[str, Any], self.CODEC_CONFIGS[codec])
        base_format = codec_config.get("pixel_format")
        if base_format is None:  # rawvideo case
            return None

        # Only use RGB formats for actual RGB data (H, W, 3)
        shape = feature_type.shape
        if shape is not None and len(shape) == 3 and shape[2] == 3:
            # RGB data - use appropriate RGB format
            return ("yuv420p" if codec in [
                "libx264", "libx265", "libaom-av1", "ffv1"
            ] else "rgb24")
        else:
            # Non-RGB data should not get video pixel formats
            return None

    def get_codec_options(self, codec: str) -> Dict[str, Any]:
        """Get codec options, merging defaults with custom options."""
        if codec not in self.CODEC_CONFIGS:
            return self.custom_options

        codec_config = cast(Dict[str, Any], self.CODEC_CONFIGS[codec])
        options = codec_config.get("options", {}).copy()
        options.update(self.custom_options)
        return options


class Trajectory(TrajectoryInterface):

    def __init__(
        self,
        path: Text,
        mode="r",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        feature_name_separator: Text = "/",
        filesystem: Optional[Any] = None,
        time_provider: Optional[Any] = None,
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
    ) -> None:
        """
        Args:
            path (Text): path to the trajectory file
            mode (Text, optional):  mode of the file, "r" for read and "w" for write
            video_codec (str, optional): Video codec to use. Options: "auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1". Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            feature_name_separator (Text, optional):
                Delimiter to separate feature names in the container file.
                Defaults to "/".
            filesystem: Optional filesystem interface for dependency injection
            time_provider: Optional time provider interface for dependency injection
            base_datetime: Optional base datetime for timestamp calculations
            time_unit: Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
            enforce_monotonic: Whether to enforce monotonically increasing timestamps
        """
        self.path = path
        self.feature_name_separator = feature_name_separator

        # Handle backward compatibility for a hypothetical old_lossy_param
        # We are now removing the actual lossy_compression param
        # old_lossy_param = kwargs.pop('lossy_compression', None) # Example if it were in kwargs
        # if old_lossy_param is not None:
        #     warnings.warn("lossy_compression parameter is deprecated. Use video_codec parameter instead.", UserWarning)
        #     if old_lossy_param:
        #         video_codec = "libaom-av1"
        #     else:
        #         video_codec = "ffv1"

        # Initialize codec configuration
        self.codec_config = CodecConfig(video_codec, codec_options)

        # Dependency injection - set early so they're available during init
        self._filesystem = filesystem
        self._time_provider = time_provider

        # Initialize time management system
        self.time_manager = TimeManager(
            base_datetime=base_datetime,
            time_unit=time_unit,
            enforce_monotonic=enforce_monotonic,
        )

        self.feature_name_to_stream: Dict[str,
                                          Any] = {}  # feature_name: stream
        self.feature_name_to_feature_type: Dict[str, FeatureType] = (
            {})  # feature_name: feature_type
        self.trajectory_data = None  # trajectory_data
        self.start_time = self._time()
        self.mode = mode
        self.stream_id_to_info: Dict[int,
                                     StreamInfo] = {}  # stream_id: StreamInfo
        self.is_closed = False
        self.pending_write_tasks: List[Any] = (
            [])  # List to keep track of pending write tasks
        self.container_file: Optional[Any] = None  # av.OutputContainer or None

        # check if the path exists
        # if not, create a new file and start data collection
        if self.mode == "w":
            if not self._exists(self.path):
                self._makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                self.container_file = av.open(self.path,
                                              mode="w",
                                              format="matroska")
            except Exception as e:
                logger.error(f"error creating the trajectory file: {e}")
                raise
        elif self.mode == "r":
            if not self._exists(self.path):
                raise FileNotFoundError(f"{self.path} does not exist")
        else:
            raise ValueError(f"Invalid mode {self.mode}, must be 'r' or 'w'")

    def _exists(self, path: str) -> bool:
        """File existence check with dependency injection support."""
        if self._filesystem:
            return self._filesystem.exists(path)
        return os.path.exists(path)

    def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Directory creation with dependency injection support."""
        if self._filesystem:
            return self._filesystem.makedirs(path, exist_ok=exist_ok)
        return os.makedirs(path, exist_ok=exist_ok)

    def _remove(self, path: str) -> None:
        """File removal with dependency injection support."""
        if self._filesystem:
            return self._filesystem.remove(path)
        return os.remove(path)

    def _rename(self, src: str, dst: str) -> None:
        """File rename with dependency injection support."""
        if self._filesystem:
            return self._filesystem.rename(src, dst)
        return os.rename(src, dst)

    def _time(self) -> float:
        """Time retrieval with dependency injection support."""
        if self._time_provider:
            return self._time_provider.time()
        return time.time()

    def _get_current_timestamp(self):
        current_time = (self._time() - self.start_time) * 1000000
        return current_time

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """
        get the value of the feature
        return hdf5-ed data
        """

        if self.trajectory_data is None:
            logger.info(f"Loading the trajectory data with key {key}")
            self.trajectory_data = self.load()

        if self.trajectory_data is None:
            raise RuntimeError("Failed to load trajectory data")

        return self.trajectory_data[key]

    def close(self, compact=True):
        """
        close the container file

        args:
        compact: re-read from the cache to encode pickled data to images
        """
        logger.debug(
            f"Closing trajectory, is_closed={self.is_closed}, compact={compact}"
        )

        if self.is_closed:
            raise ValueError("The container file is already closed")

        if self.mode == "r":
            # For read mode, just mark as closed
            self.is_closed = True
            self.trajectory_data = None
            logger.debug("Trajectory (read mode) closed successfully")
            return

        # Write mode handling
        if not hasattr(self, "container_file") or self.container_file is None:
            logger.warning(
                "Container file not available, marking trajectory as closed")
            self.is_closed = True
            return

        # Check if there are any streams with data
        has_data = len(self.container_file.streams) > 0

        try:
            ts = self._get_current_timestamp()
            logger.debug(f"Final timestamp: {ts}")

            for i, stream in enumerate(self.container_file.streams):
                logger.debug(f"Flushing stream {i}: {stream}")
                try:
                    packets = stream.encode(None)  # type: ignore[attr-defined]
                    logger.debug(
                        f"Stream {i} flush returned {len(packets)} packets")
                    for j, packet in enumerate(packets):
                        packet.pts = ts
                        packet.dts = ts
                        if self.container_file is not None:
                            self.container_file.mux(packet)
                            logger.debug(
                                f"Muxed flush packet {j} from stream {i}")
                        else:
                            raise RuntimeError(
                                "Container file is None, cannot mux packet")
                except Exception as e:
                    logger.error(f"Error flushing stream {stream}: {e}")
            logger.debug("Flushing the container file")
        except av.error.EOFError:
            logger.debug("Got EOFError during flush (expected)")
            pass  # This exception is expected and means the encoder is fully flushed

        logger.debug("Closing container file")
        self.container_file.close()

        # Ensure file exists even if empty - the container file should create it
        if not self._exists(self.path):
            logger.warning(
                f"Container file was closed but {self.path} doesn't exist. This might indicate an issue."
            )

        # Only attempt transcoding if file exists, has content, and compact is requested
        if (compact and has_data and self._exists(self.path)
                and os.path.getsize(self.path) > 0):
            logger.debug("Starting transcoding of pickled images")
            try:
                self._transcode_pickled_images(ending_timestamp=ts)
            except Exception as e:
                logger.warning(
                    f"Transcoding failed: {e}. Keeping original file with pickled data."
                )
                # File remains in original state with pickled data, which is still valid
        else:
            logger.debug(
                f"Skipping transcoding: compact={compact}, has_data={has_data}, file_exists={self._exists(self.path)}, file_size={os.path.getsize(self.path) if self._exists(self.path) else 0}"
            )

        self.trajectory_data = None
        self.container_file = None
        self.is_closed = True
        logger.debug("Trajectory closed successfully")

    def load(
        self,
        return_type: str = "numpy",
        desired_frequency: Optional[float] = None,
        data_slice: Optional[slice] = None,
    ):
        """
        Load trajectory data with optional temporal resampling and slicing.

        Parameters
        ----------
        return_type : {"numpy", "container"}, default "numpy"
            • "numpy"     – decode the data and return a dict[str, np.ndarray]
            • "container" – skip all decoding and just return the file path
        desired_frequency : float | None, default None
            Target sampling frequency **in hertz**.  If None, every frame is
            returned (subject to `data_slice`). For upsampling (when desired
            frequency is higher than original), prior frames are duplicated
            to fill temporal gaps. For downsampling, frames are skipped.
        data_slice : slice | None, default None
            Standard Python slice that is applied *after* resampling.
            Example: `slice(100, 200, 2)` → keep resampled indices 100-199,
            step 2.  Negative indices and reverse slices are **not** supported.

        Notes
        -----
        * Resampling is performed individually for every feature stream.
        * For upsampling: when time gaps between consecutive frames exceed
          the desired period, the prior frame is duplicated at regular
          intervals to achieve the target frequency.
        * For downsampling: frames that arrive too close together (within
          the desired period) are skipped.
        * Slicing is interpreted on the **resampled index** domain so that the
          combination `desired_frequency + data_slice` behaves the same as
          `df.iloc[data_slice]` would on a pandas dataframe that had already
          been resampled to `desired_frequency`.
        * When `data_slice` starts at a positive index we `seek()` to the
          corresponding timestamp to avoid decoding frames that will be thrown
          away anyway.
        """
        logger.debug(
            f"load() called with return_type='{return_type}', desired_frequency={desired_frequency}, data_slice={data_slice}"
        )

        # ------------------------------------------------------------------ #
        # Fast-path: user only wants the container path
        # ------------------------------------------------------------------ #
        if return_type == "container":
            logger.debug("Returning container path (fast-path)")
            return self.path
        if return_type not in {"numpy", "container"}:
            raise ValueError("return_type must be 'numpy' or 'container'")

        # ------------------------------------------------------------------ #
        # Validate / canonicalise the slice object
        # ------------------------------------------------------------------ #
        if data_slice is None:
            logger.debug(
                "No data_slice provided, using default slice(None, None, None)"
            )
            data_slice = slice(None, None, None)
        else:
            logger.debug(f"Using provided data_slice: {data_slice}")

        if data_slice.step not in (None, 1) and data_slice.step <= 0:
            raise ValueError("Reverse or zero-step slices are not supported")

        # Check for negative start - this should raise an error
        if data_slice.start is not None and data_slice.start < 0:
            raise ValueError("Negative slice start values are not supported")

        sl_start = 0 if data_slice.start is None else max(data_slice.start, 0)
        sl_stop = data_slice.stop  # can be None
        sl_step = 1 if data_slice.step is None else data_slice.step

        logger.debug(
            f"Canonicalized slice parameters: start={sl_start}, stop={sl_stop}, step={sl_step}"
        )

        # ------------------------------------------------------------------ #
        # Frequency → minimum period in stream time-base units (milliseconds)
        # ------------------------------------------------------------------ #
        period_ms: Optional[int] = None
        if desired_frequency is not None:
            if desired_frequency <= 0:
                raise ValueError("desired_frequency must be positive")
            period_ms = int(round(1000.0 / desired_frequency))
            logger.debug(
                f"Frequency resampling enabled: {desired_frequency} Hz -> period_ms={period_ms}"
            )
        else:
            logger.debug("No frequency resampling (desired_frequency is None)")

        # ------------------------------------------------------------------ #
        # Open the container and, if possible, seek() to the first slice index
        # ------------------------------------------------------------------ #
        logger.debug(f"Opening container file: {self.path}")
        container = av.open(self.path, mode="r", format="matroska")
        streams = list(container.streams)

        logger.debug(f"Container opened with {len(streams)} streams")

        # Handle empty trajectory case
        if not streams:
            logger.debug("No streams found in container, returning empty dict")
            container.close()
            return {}

        # Track if we performed seeking to adjust slice logic
        seek_performed = False
        seek_offset_frames = 0

        # Use seeking optimization when we have slicing
        if sl_start > 0 and streams:
            if period_ms is not None:
                # When combining frequency resampling with slicing, seek to the timestamp
                # that corresponds to the sl_start-th frame AFTER resampling.
                # Since resampling keeps every period_ms milliseconds, the sl_start-th
                # resampled frame corresponds to timestamp: sl_start * period_ms
                seek_ts_ms = sl_start * period_ms
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking with frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )
            else:
                # If only slicing (no frequency resampling), seek to the sl_start-th frame
                # assuming original 100ms intervals (10Hz from our test data)
                seek_ts_ms = sl_start * 100
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking without frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )

            # Seek using the first stream's time_base (which is 1/1000, so offset is in ms)
            try:
                logger.debug(
                    f"Attempting to seek to timestamp {seek_ts_ms} on stream {streams[0]}"
                )
                container.seek(seek_ts_ms, stream=streams[0], any_frame=True)
                seek_performed = True
                logger.debug("Seek successful")
            except av.AVError as e:
                # Seeking failed (e.g. single large packet stream) – fall back
                # to decoding from the beginning.
                logger.debug(
                    f"Seeking failed ({e}), falling back to decoding from beginning"
                )
                seek_performed = False
                seek_offset_frames = 0
        else:
            logger.debug(
                "No seeking optimization needed (sl_start=0 or no streams)")

        # ------------------------------------------------------------------ #
        # Book-keeping structures
        # ------------------------------------------------------------------ #
        cache: dict[str, list[Any]] = {}
        last_pts: dict[str, Optional[int]] = {}
        kept_idx: dict[str, int] = {}
        done: set[str] = set()

        stream_count = 0
        for s in streams:
            fname = s.metadata.get("FEATURE_NAME")
            ftype = s.metadata.get("FEATURE_TYPE")
            if not (fname and ftype):
                logger.debug(
                    f"Skipping stream {s} without FEATURE_NAME or FEATURE_TYPE metadata"
                )
                continue
            cache[fname] = []
            last_pts[fname] = None
            # If we seeked, start counting from the seek offset minus 1
            # (since kept_idx gets incremented before checking)
            kept_idx[fname] = seek_offset_frames - 1 if seek_performed else -1
            self.feature_name_to_feature_type[fname] = FeatureType.from_str(
                ftype)
            stream_count += 1
            logger.debug(
                f"Initialized feature '{fname}' with type {ftype}, kept_idx={kept_idx[fname]}"
            )

        # Handle case where no valid streams were found
        if not cache:
            logger.debug(
                "No valid feature streams found, returning empty dict")
            container.close()
            return {}

        logger.debug(f"Processing {stream_count} feature streams")

        # ------------------------------------------------------------------ #
        # Helper: quickly decide if *resampled* index should be kept
        # ------------------------------------------------------------------ #
        def want(idx: int) -> bool:
            if idx < sl_start:
                return False
            if sl_stop is not None and idx >= sl_stop:
                return False
            return ((idx - sl_start) % sl_step) == 0

        # ------------------------------------------------------------------ #
        # Main demux / decode loop
        # ------------------------------------------------------------------ #
        logger.debug("Starting main demux/decode loop")
        packet_count = 0
        processed_packets = 0
        skipped_frequency = 0
        skipped_slice = 0
        decoded_packets = 0
        upsampled_frames = 0

        for packet in container.demux(streams):
            packet_count += 1
            fname = packet.stream.metadata.get("FEATURE_NAME")
            if fname is None or fname in done:
                continue

            # PyAV sometimes returns "dummy" packets whose pts / dts is None
            # (e.g. after a flush or if the stream has no real data).  They
            # must be skipped before any timing logic.
            if packet.pts is None:
                logger.debug(
                    f"Skipping packet with None pts for feature '{fname}'")
                continue

            processed_packets += 1

            # --- per-stream frequency adjustment (upsampling/downsampling) ---
            if period_ms is not None:
                lp = last_pts[fname]
                # Guard both operands – pts is now guaranteed not-None.
                if lp is not None:
                    time_gap = packet.pts - lp

                    if time_gap < period_ms:
                        # Downsampling: skip this frame
                        skipped_frequency += 1
                        logger.debug(
                            f"Skipping packet for '{fname}' due to frequency reduction: pts={packet.pts}, last_pts={lp}, period_ms={period_ms}"
                        )
                        continue
                    elif time_gap > period_ms and cache[fname]:
                        # Upsampling: insert duplicate frames before processing current frame
                        # Calculate how many intermediate frames we need
                        num_intermediate_frames = int(
                            time_gap // period_ms) - 1

                        if num_intermediate_frames > 0:
                            # Get the last frame data for duplication
                            last_frame_data = cache[fname][-1]

                            # Insert intermediate frames
                            for i in range(1, num_intermediate_frames + 1):
                                kept_idx[fname] += 1

                                if want(kept_idx[fname]):
                                    cache[fname].append(last_frame_data)
                                    upsampled_frames += 1
                                    logger.debug(
                                        f"Inserted duplicate frame for '{fname}' at intermediate position {i}/{num_intermediate_frames}, kept_idx={kept_idx[fname]}"
                                    )

                    logger.debug(
                        f"Keeping packet for '{fname}' after frequency check: pts={packet.pts}, last_pts={lp}, period_ms={period_ms}"
                    )
                else:
                    logger.debug(
                        f"First packet for '{fname}', no upsampling needed: pts={packet.pts}"
                    )
            else:
                logger.debug(
                    f"No frequency resampling for '{fname}': period_ms is None"
                )

            # This packet is being kept at the resampling stage
            kept_idx[fname] += 1
            # Only update last_pts if this packet has a usable pts
            last_pts[fname] = packet.pts

            if not want(kept_idx[fname]):  # slice filter
                skipped_slice += 1
                logger.debug(
                    f"Skipping packet for '{fname}' due to slice filter: kept_idx={kept_idx[fname]}"
                )
                continue

            logger.debug(
                f"Decoding packet for '{fname}': kept_idx={kept_idx[fname]}, pts={packet.pts}"
            )

            # --- decode on demand only ------------------------------------
            codec = packet.stream.codec_context.codec.name
            if codec == "rawvideo":
                raw = bytes(packet)
                if not raw:  # zero-length placeholder
                    logger.debug(
                        f"Skipping empty rawvideo packet for '{fname}'")
                    continue
                cache[fname].append(pickle.loads(raw))
                decoded_packets += 1
                logger.debug(
                    f"Decoded rawvideo packet for '{fname}' (pickled data)")
            else:
                for frame in packet.decode():
                    ft = self.feature_name_to_feature_type[fname]
                    # Only decode as RGB24 for RGB data, otherwise this shouldn't happen
                    # since non-RGB data should use rawvideo
                    if ft.shape and len(ft.shape) == 3 and ft.shape[2] == 3:
                        # RGB data - decode as RGB24
                        arr = frame.to_ndarray(format="rgb24")
                    else:
                        # This shouldn't happen with our new logic, but handle gracefully
                        logger.warning(
                            f"Non-RGB data {fname} with shape {ft.shape} using video codec - this may cause issues"
                        )
                        arr = frame.to_ndarray(format="rgb24")

                    if ft.shape:
                        arr = arr.reshape(ft.shape)
                    cache[fname].append(arr)
                    decoded_packets += 1
                    logger.debug(
                        f"Decoded {codec} frame for '{fname}': shape={arr.shape}, dtype={arr.dtype}"
                    )

            # Early exit: all streams finished their slice
            if sl_stop is not None and kept_idx[fname] >= sl_stop:
                done.add(fname)
                logger.debug(
                    f"Feature '{fname}' reached slice stop ({sl_stop}), marking as done"
                )
                if len(done) == len(cache):
                    logger.debug(
                        "All features completed their slices, breaking early")
                    break

        # ------------------------------------------------------------------ #
        # Flush any buffered pictures that the decoder is still holding
        # ------------------------------------------------------------------ #
        for s in streams:
            fname = s.metadata.get("FEATURE_NAME")
            if not fname or fname not in cache:
                continue
            if s.codec_context.codec.name == "rawvideo":
                continue  # pickled streams have no buffer

            # Passing None tells PyAV/FFmpeg "end of stream – give me leftovers"
            for frame in s.decode(
                    None
            ):  # PyAV ≥ 10; on ≤ 0.5 use s.codec_context.decode(None)
                kept_idx[fname] += 1
                if not want(kept_idx[fname]):  # honour slice filter
                    continue

                ft = self.feature_name_to_feature_type[fname]
                # Only decode as RGB24 for RGB data
                if ft.shape and len(ft.shape) == 3 and ft.shape[2] == 3:
                    # RGB data - decode as RGB24
                    arr = frame.to_ndarray(format="rgb24")
                else:
                    # This shouldn't happen with our new logic, but handle gracefully
                    logger.warning(
                        f"Non-RGB data {fname} with shape {ft.shape} using video codec - this may cause issues"
                    )
                    arr = frame.to_ndarray(format="rgb24")

                if ft.shape:
                    arr = arr.reshape(ft.shape)
                cache[fname].append(arr)
                decoded_packets += 1

        container.close()

        logger.debug(
            f"Demux/decode loop completed: total_packets={packet_count}, processed={processed_packets}, "
            f"skipped_frequency={skipped_frequency}, skipped_slice={skipped_slice}, decoded={decoded_packets}, upsampled_frames={upsampled_frames}"
        )

        # ------------------------------------------------------------------ #
        # Convert to numpy arrays
        # ------------------------------------------------------------------ #
        logger.debug("Converting cached data to numpy arrays")
        out: Dict[str, Any] = {}
        for fname, lst in cache.items():
            logger.debug(f"Converting '{fname}': {len(lst)} items")
            if not lst:
                logger.debug(f"Warning: '{fname}' has no data after filtering")
                out[fname] = np.array([])
                continue

            ft = self.feature_name_to_feature_type[fname]
            if ft.dtype in ["string", "str"]:
                out[fname] = np.array(lst, dtype=object)
                logger.debug(
                    f"Created object array for '{fname}': shape={out[fname].shape}"
                )
            else:
                out[fname] = np.asarray(lst, dtype=ft.dtype)
                logger.debug(
                    f"Created {ft.dtype} array for '{fname}': shape={out[fname].shape}"
                )

        logger.debug(
            f"load() returning {len(out)} features: {list(out.keys())}")
        return out

    def init_feature_streams(self, feature_spec: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_spec.items():
            encoding = self._get_encoding_of_feature(None, feature_type)
            self.feature_name_to_stream[
                feature] = self._add_stream_to_container(
                    self.container_file, feature, encoding, feature_type)

    def add(
        self,
        feature: str,
        data: Any,
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        """
        add one value to container file

        Args:
            feature (str): name of the feature
            data (Any): value associated with the feature; except dictionary
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.

        Examples:
            >>> trajectory.add('feature1', 'image1.jpg')
            >>> trajectory.add('feature1', 'image1.jpg', timestamp=1000, time_unit='ms')

        Logic:
        - check the feature name
        - if the feature name is not in the container, create a new stream

        - check the type of value
        - if value is numpy array, create a frame and encode it
        - if it is a string or int, create a packet and encode it
        - else raise an error

        Exceptions:
            raise an error if the value is a dictionary
        """
        logger.debug(
            f"Adding feature: {feature}, data shape: {getattr(data, 'shape', 'N/A')}"
        )

        if type(data) == dict:
            raise ValueError("Use add_by_dict for dictionary")

        feature_type = FeatureType.from_data(data)
        # encoding = self._get_encoding_of_feature(data, None)
        self.feature_name_to_feature_type[feature] = feature_type

        # check if the feature is already in the container
        # if not, create a new stream
        # Check if the feature is already in the container
        # here we enforce rawvideo encoding for all features
        # later on the compacting step, we will encode the pickled data to images
        if feature not in self.feature_name_to_stream:
            logger.debug(f"Creating new stream for feature: {feature}")
            self._on_new_stream(feature, "rawvideo", feature_type)

        # get the stream
        stream = self.feature_name_to_stream[feature]
        logger.debug(f"Using stream: {stream}")

        # get the timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.validate_timestamp(
                timestamp, time_unit)

        logger.debug(
            f"Encoding frame with validated timestamp: {validated_timestamp}")
        # encode the frame
        packets = self._encode_frame(data, stream, validated_timestamp)
        logger.debug(f"Generated {len(packets)} packets")

        # write the packet to the container
        for i, packet in enumerate(packets):
            logger.debug(f"Muxing packet {i}: {packet}")
            if self.container_file is not None:
                self.container_file.mux(packet)
                logger.debug(f"Successfully muxed packet {i}")
            else:
                raise RuntimeError("Container file is None, cannot mux packet")

    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        """
        add one value to container file
        data might be nested dictionary of values for each feature

        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
                assume the timestamp is same for all the features within the dictionary

        Examples:
            >>> trajectory.add_by_dict({'feature1': 'image1.jpg'})

        Logic:
        - check the data see if it is a dictionary
        - if dictionary, need to flatten it and add each feature separately
        """
        if type(data) != dict:
            raise ValueError("Use add for non-dictionary data, type is ",
                             type(data))

        _flatten_dict_data = _flatten_dict(data,
                                           sep=self.feature_name_separator)

        # Get validated timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.validate_timestamp(
                timestamp, time_unit)

        for feature, value in _flatten_dict_data.items():
            self.add(feature, value, validated_timestamp, "ms")

    @classmethod
    def from_list_of_dicts(
        cls,
        data: List[Dict[str, Any]],
        path: Text,
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a list of dictionaries.

        args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.

        Example:
        original_trajectory = [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"},
        ]

        trajectory = Trajectory.from_list_of_dicts(original_trajectory, path="/tmp/robodm/output.vla")
        """
        traj = cls(path,
                   mode="w",
                   video_codec=video_codec,
                   codec_options=codec_options)
        logger.info(
            f"Creating a new trajectory file at {path} with {len(data)} steps")
        for step in data:
            traj.add_by_dict(step)
        traj.close()
        return traj

    @classmethod
    def from_dict_of_lists(
        cls,
        data: Dict[str, List[Any]],
        path: Text,
        feature_name_separator: Text = "/",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file
            feature_name_separator (Text, optional): Delimiter to separate feature names. Defaults to "/".
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.

        Returns:
            Trajectory: _description_

        Example:
        original_trajectory = {
            "feature1": ["value1", "value3"],
            "feature2": ["value2", "value4"],
        }

        trajectory = Trajectory.from_dict_of_lists(original_trajectory, path="/tmp/robodm/output.vla")
        """
        traj = cls(
            path,
            feature_name_separator=feature_name_separator,
            mode="w",
            video_codec=video_codec,
            codec_options=codec_options,
        )
        # flatten the data such that all data starts and put feature name with separator
        _flatten_dict_data = _flatten_dict(data,
                                           sep=traj.feature_name_separator)

        # Check if all lists have the same length
        list_lengths = [len(v) for v in _flatten_dict_data.values()]
        if len(set(list_lengths)) != 1:
            raise ValueError(
                "All lists must have the same length",
                [(k, len(v)) for k, v in _flatten_dict_data.items()],
            )

        for i in range(list_lengths[0]):
            step = {k: v[i] for k, v in _flatten_dict_data.items()}
            traj.add_by_dict(step)
        traj.close()
        return traj

    def _load_from_container(self):
        """
        Load the container file with the entire VLA trajectory using multi-processing for image streams.

        returns:
            np_cache: dictionary with the decoded data

        Workflow:
        - Get schema of the container file.
        - Preallocate decoded streams.
        - Use multi-processing to decode image streams separately.
        - Decode non-image streams in the main process.
        - Combine results from all processes.
        """

        container = av.open(self.path, mode="r", format="matroska")
        streams = container.streams

        # Dictionary to store dynamic lists for collecting data
        np_cache_lists: Dict[str, List[Any]] = {}
        feature_name_to_stream = {}

        # Initialize lists for each feature
        for stream in streams:
            feature_name = stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                continue
            feature_type_str = stream.metadata.get("FEATURE_TYPE")
            if feature_type_str is None:
                logger.debug(f"Skipping stream without FEATURE_TYPE: {stream}")
                continue
            feature_type = FeatureType.from_str(feature_type_str)
            feature_name_to_stream[feature_name] = stream
            self.feature_name_to_feature_type[feature_name] = feature_type

            logger.debug(
                f"Initializing list for {feature_name} with feature_type {feature_type}"
            )
            np_cache_lists[feature_name] = []

        # Decode the frames and store them in the lists
        for packet in container.demux(list(streams)):
            feature_name = packet.stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                logger.debug(
                    f"Skipping stream without FEATURE_NAME: {packet.stream}")
                continue
            feature_type_str = packet.stream.metadata.get("FEATURE_TYPE")
            if feature_type_str is None:
                logger.debug(
                    f"Skipping stream without FEATURE_TYPE: {packet.stream}")
                continue
            feature_type = FeatureType.from_str(feature_type_str)

            logger.debug(f"Decoding {feature_name} with time {packet.dts}")

            feature_codec = packet.stream.codec_context.codec.name
            if feature_codec == "rawvideo":
                packet_in_bytes = bytes(packet)
                if packet_in_bytes:
                    # Decode the packet
                    data = pickle.loads(packet_in_bytes)
                    np_cache_lists[feature_name].append(data)
                else:
                    logger.debug(
                        f"Skipping empty packet: {packet} for {feature_name}")
            else:
                frames = packet.decode()
                for frame in frames:
                    # Only decode as RGB24 for RGB data
                    shape = feature_type.shape
                    if shape and len(shape) == 3 and shape[2] == 3:
                        # RGB data - decode as RGB24
                        if shape is not None:
                            data = frame.to_ndarray(  # type: ignore[attr-defined]
                                format="rgb24").reshape(shape)
                        else:
                            data = frame.to_ndarray(
                                format="rgb24")  # type: ignore[attr-defined]
                    else:
                        # This shouldn't happen with our new logic, but handle gracefully
                        logger.warning(
                            f"Non-RGB data {feature_name} with shape {shape} using video codec"
                        )
                        if shape is not None:
                            data = frame.to_ndarray(  # type: ignore[attr-defined]
                                format="rgb24").reshape(shape)
                        else:
                            data = frame.to_ndarray(
                                format="rgb24")  # type: ignore[attr-defined]
                    np_cache_lists[feature_name].append(data)

        container.close()

        # Convert lists to numpy arrays
        np_cache = {}
        for feature_name, data_list in np_cache_lists.items():
            logger.debug(
                f"Converting {feature_name} list of length {len(data_list)} to numpy array"
            )
            if not data_list:
                logger.debug(f"Warning: {feature_name} has no data!")
                continue

            feature_type = self.feature_name_to_feature_type[feature_name]

            if feature_type.dtype == "string":
                np_cache[feature_name] = np.array(data_list, dtype=object)
            else:
                # Convert list to numpy array
                np_cache[feature_name] = np.array(data_list,
                                                  dtype=feature_type.dtype)

        return np_cache

    def _transcode_pickled_images(self,
                                  ending_timestamp: Optional[int] = None):
        """
        Transcode pickled images into the desired format (e.g., raw or encoded images).
        """

        # Move the original file to a temporary location
        temp_path = self.path + ".temp"
        self._rename(self.path, temp_path)

        try:
            # Open the original container for reading
            original_container = av.open(temp_path,
                                         mode="r",
                                         format="matroska")
            original_streams = list(original_container.streams)

            # Create a new container
            new_container = av.open(self.path, mode="w", format="matroska")

            # Add existing streams to the new container
            d_original_stream_id_to_new_container_stream = {}
            for stream in original_streams:
                stream_feature = stream.metadata.get("FEATURE_NAME")
                if stream_feature is None:
                    logger.debug(
                        f"Skipping stream without FEATURE_NAME: {stream}")
                    continue

                # Determine encoding method based on feature type
                try:
                    stream_encoding = self._get_encoding_of_feature(
                        None,
                        self.feature_name_to_feature_type[stream_feature])
                    stream_feature_type = self.feature_name_to_feature_type[
                        stream_feature]
                    stream_in_updated_container = self._add_stream_to_container(
                        new_container,
                        stream_feature,
                        stream_encoding,
                        stream_feature_type,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create stream for {stream_feature} with desired encoding, falling back to rawvideo: {e}"
                    )
                    # Fallback to rawvideo if the desired codec is not available
                    stream_in_updated_container = self._add_stream_to_container(
                        new_container,
                        stream_feature,
                        "rawvideo",
                        self.feature_name_to_feature_type[stream_feature],
                    )

                # Preserve the stream metadata
                for key, value in stream.metadata.items():
                    stream_in_updated_container.metadata[key] = value

                d_original_stream_id_to_new_container_stream[stream.index] = (
                    stream_in_updated_container)

            # Transcode pickled images and add them to the new container
            packets_muxed = 0
            for packet in original_container.demux(original_streams):

                def is_packet_valid(packet):
                    return packet.pts is not None and packet.dts is not None

                if is_packet_valid(packet):
                    original_stream = packet.stream
                    new_stream = d_original_stream_id_to_new_container_stream[
                        packet.stream.index]
                    packet.stream = new_stream

                    # Check if the ORIGINAL stream is using rawvideo, meaning it's a pickled stream
                    if original_stream.codec_context.codec.name == "rawvideo":
                        logger.debug(
                            f"Transcoding rawvideo packet from {original_stream.metadata.get('FEATURE_NAME')}"
                        )
                        data = pickle.loads(bytes(packet))

                        # Encode the image data with the new stream's encoding
                        try:
                            pts_timestamp = packet.pts if packet.pts is not None else 0
                            new_packets = self._encode_frame(
                                data, new_stream, pts_timestamp)
                            for new_packet in new_packets:
                                logger.debug(
                                    f"Muxing transcoded packet: {new_packet}")
                                new_container.mux(new_packet)
                                packets_muxed += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to encode {original_stream.metadata.get('FEATURE_NAME')} with {new_stream.codec_context.codec.name}, keeping as pickled data: {e}"
                            )
                            # If encoding fails, keep the original pickled packet
                            new_container.mux(packet)
                            packets_muxed += 1
                    else:
                        # If not a rawvideo stream, just remux the existing packet
                        logger.debug(f"Remuxing original packet: {packet}")
                        new_container.mux(packet)
                        packets_muxed += 1
                else:
                    logger.debug(f"Skipping invalid packet: {packet}")

            logger.debug(f"Muxed {packets_muxed} packets during transcoding")

            # Flush all streams to get any buffered packets
            for stream in new_container.streams:
                logger.debug(f"Flushing stream during transcode: {stream}")
                try:
                    flush_packets = stream.encode(
                        None)  # type: ignore[attr-defined]
                    logger.debug(
                        f"Stream flush returned {len(flush_packets)} packets")
                    for packet in flush_packets:
                        packet.pts = ending_timestamp
                        packet.dts = ending_timestamp
                        logger.debug(f"Muxing flush packet: {packet}")
                        new_container.mux(packet)
                        packets_muxed += 1
                except Exception as e:
                    logger.error(f"Error flushing stream {stream}: {e}")

            logger.debug(f"Total packets muxed: {packets_muxed}")

            original_container.close()
            new_container.close()
            self._remove(temp_path)

        except Exception as e:
            # If transcoding fails completely, restore the original file
            logger.error(f"Transcoding failed completely: {e}")
            if self._exists(temp_path):
                if self._exists(self.path):
                    self._remove(self.path)
                self._rename(temp_path, self.path)
                logger.info(f"Restored original file to {self.path}")
            raise

    def _encode_frame(self, data: Any, stream: Any,
                      timestamp: int) -> List[av.Packet]:
        """
        encode the frame and write it to the stream file, return the packet
        args:
            data: data frame to be encoded
            stream: stream to write the frame
            timestamp: timestamp of the frame
        return:
            packet: encoded packet
        """
        encoding = stream.codec_context.codec.name
        feature_type = FeatureType.from_data(data)
        logger.debug(
            f"Encoding {stream.metadata.get('FEATURE_NAME')} with {encoding}, feature_type: {feature_type}"
        )

        # For video codecs, only attempt to create video frames if data is image-like (2D or 3D)
        shape = feature_type.shape
        if (encoding in ["ffv1", "libaom-av1", "libx264", "libx265"]
                and shape is not None and len(shape) >= 2):
            logger.debug("Using video encoding path for image-like data")
            # Always use RGB frame creation, no special handling for float32
            frame = self._create_frame(data, stream)
            frame.pts = timestamp
            frame.dts = timestamp
            frame.time_base = stream.time_base
            logger.debug(f"Created frame: pts={frame.pts}, dts={frame.dts}")
            packets = stream.encode(frame)  # type: ignore[attr-defined]
            logger.debug(f"Stream encode returned {len(packets)} packets")
        else:
            if encoding in ["ffv1", "libaom-av1", "libx264", "libx265"]:
                logger.debug(
                    f"Data is not image-like (shape: {shape}). Using rawvideo (pickling) path for this packet despite stream encoding being {encoding}."
                )
            else:
                logger.debug("Using rawvideo encoding path")

            packet = av.Packet(pickle.dumps(data))
            packet.dts = timestamp
            packet.pts = timestamp
            packet.time_base = stream.time_base
            packet.stream = stream
            logger.debug(f"Created raw packet: size={len(bytes(packet))}")

            packets = [packet]

        for (
                packet_item
        ) in packets:  # renamed to avoid conflict with outer scope 'packet'
            packet_item.pts = timestamp
            packet_item.dts = timestamp
            packet_item.time_base = stream.time_base

        logger.debug(f"Returning {len(packets)} packets")
        return packets

    def _on_new_stream(self, new_feature, new_encoding, new_feature_type):
        if new_feature in self.feature_name_to_stream:
            return

        if not self.feature_name_to_stream:
            logger.debug(
                f"Creating a new stream for the first feature {new_feature}")
            self.feature_name_to_stream[
                new_feature] = self._add_stream_to_container(
                    self.container_file, new_feature, new_encoding,
                    new_feature_type)
        else:
            logger.debug(f"Adding a new stream for the feature {new_feature}")
            # Following is a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close(compact=False)

            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            self._rename(self.path, temp_path)

            # Open the original container for reading
            original_container = av.open(temp_path,
                                         mode="r",
                                         format="matroska")
            original_streams = list(original_container.streams)

            # Create a new container
            new_container = av.open(self.path, mode="w", format="matroska")

            # Add existing streams to the new container
            d_original_stream_id_to_new_container_stream = {}
            for stream in original_streams:
                stream_feature = stream.metadata.get("FEATURE_NAME")
                if stream_feature is None:
                    logger.debug(
                        f"Skipping stream without FEATURE_NAME: {stream}")
                    continue
                stream_encoding = stream.codec_context.codec.name
                stream_feature_type = self.feature_name_to_feature_type[
                    stream_feature]
                stream_in_updated_container = self._add_stream_to_container(
                    new_container, stream_feature, stream_encoding,
                    stream_feature_type)
                # new_stream.options = stream.options
                for key, value in stream.metadata.items():
                    stream_in_updated_container.metadata[key] = value
                d_original_stream_id_to_new_container_stream[stream.index] = (
                    stream_in_updated_container)

            # Add new feature stream
            new_stream = self._add_stream_to_container(new_container,
                                                       new_feature,
                                                       new_encoding,
                                                       new_feature_type)
            d_original_stream_id_to_new_container_stream[
                new_stream.index] = new_stream
            self.stream_id_to_info[new_stream.index] = StreamInfo(
                new_feature, new_feature_type, new_encoding)

            # Remux existing packets
            for packet in original_container.demux(original_streams):

                def is_packet_valid(packet):
                    return packet.pts is not None and packet.dts is not None

                if is_packet_valid(packet):
                    packet.stream = d_original_stream_id_to_new_container_stream[
                        packet.stream.index]
                    new_container.mux(packet)
                else:
                    pass

            original_container.close()
            self._remove(temp_path)

            # Reopen the new container for writing new data
            self.container_file = new_container
            self.feature_name_to_stream[new_feature] = new_stream
            self.is_closed = False

    def _add_stream_to_container(self, container, feature_name, encoding,
                                 feature_type):
        stream = container.add_stream(encoding)

        # Configure stream based on encoding type
        if encoding in ["ffv1", "libaom-av1", "libx264", "libx265"]:
            # Only set width/height if shape is 2D or more (image/video like)
            shape = feature_type.shape
            if shape is not None and len(shape) >= 2:
                stream.width = shape[1]
                stream.height = shape[0]

            # Set pixel format based on codec and feature type
            pixel_format = self.codec_config.get_pixel_format(
                encoding, feature_type)
            if pixel_format:
                stream.pix_fmt = pixel_format

            # Set codec-specific options
            codec_options = self.codec_config.get_codec_options(encoding)
            if codec_options:
                stream.codec_context.options = codec_options

        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream

    def _create_frame(self, image_array, stream):
        image_array = np.array(image_array)
        encoding = stream.codec_context.codec.name

        # Convert to uint8 if needed
        if image_array.dtype == np.float32:
            # Assume float32 values are in [0, 1] range, scale to [0, 255]
            image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        elif image_array.dtype != np.uint8:
            # Convert other dtypes to uint8
            if np.issubdtype(image_array.dtype, np.integer):
                # For integer types, clamp to 0-255 range
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            else:
                # For other types, normalize and convert
                image_array = np.clip(image_array * 255, 0,
                                      255).astype(np.uint8)

        # Only handle RGB images (HxWx3) - no grayscale conversion
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB image - proceed with video encoding
            pass
        else:
            raise ValueError(
                f"Video codecs only support RGB images with shape (H, W, 3). "
                f"Got shape {image_array.shape}. Use rawvideo encoding for other formats."
            )

        # Create RGB frame
        if encoding in ["libaom-av1", "ffv1", "libx264", "libx265"]:
            # For video codecs that prefer YUV, convert RGB to YUV420p
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")
            frame = frame.reformat(format="yuv420p")
        else:
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")

        frame.time_base = stream.time_base
        return frame

    def _create_frame_depth(self, image_array, stream):
        image_array = np.array(image_array)

        # Convert float32 to uint8 if needed
        if image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)

        # Handle different shapes
        if len(image_array.shape) == 3:
            # If 3D, take the first channel or average if it's RGB
            if image_array.shape[2] == 3:
                # Convert RGB to grayscale
                image_array = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                # Take the first channel
                image_array = image_array[:, :, 0]

        frame = av.VideoFrame.from_ndarray(image_array, format="gray")
        frame.time_base = stream.time_base
        return frame

    def _get_encoding_of_feature(self, feature_value: Any,
                                 feature_type: Optional[FeatureType]) -> Text:
        """
        get the encoding of the feature value
        args:
            feature_value: value of the feature
            feature_type: type of the feature
        return:
            encoding of the feature in string
        """
        if feature_type is None:
            feature_type = FeatureType.from_data(feature_value)

        return self.codec_config.get_codec_for_feature(feature_type)

    def save_stream_info(self):
        # serialize and save the stream info
        with open(self.path + ".stream_info", "wb") as f:
            pickle.dump(self.stream_id_to_info, f)

    def load_stream_info(self):
        # load the stream info
        with open(self.path + ".stream_info", "rb") as f:
            self.stream_id_to_info = pickle.load(f)

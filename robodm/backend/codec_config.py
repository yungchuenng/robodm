from typing import List, Dict, Any, Optional, Tuple, cast
from fractions import Fraction
import logging
import av
from robodm.feature import FeatureType

logger = logging.getLogger(__name__)


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

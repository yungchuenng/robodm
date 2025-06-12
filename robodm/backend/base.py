from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Text, Union, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class StreamMetadata:
    """Metadata for a stream including feature name, type, and encoding"""
    feature_name: str
    feature_type: str  # Using string to avoid circular imports with FeatureType
    encoding: str
    time_base: tuple[int, int]  # Numerator, denominator for time base fraction
    additional_metadata: Dict[str, str] = None

@dataclass
class Frame:
    """Container-agnostic representation of a frame"""
    data: Union[np.ndarray, bytes]  # Raw data - either numpy array for images or bytes for pickled data
    pts: int  # Presentation timestamp
    dts: int  # Decoding timestamp
    time_base: tuple[int, int]  # Time base as (numerator, denominator)
    stream_index: int  # Index of the stream this frame belongs to
    is_keyframe: bool = False

@dataclass
class PacketInfo:
    """Container-agnostic representation of a packet"""
    data: bytes
    pts: Optional[int]
    dts: Optional[int]
    stream_index: int
    time_base: tuple[int, int]
    is_keyframe: bool = False

@dataclass 
class StreamConfig:
    """Configuration for stream creation"""
    feature_name: str
    feature_type: Any  # FeatureType object
    encoding: str
    codec_options: Optional[Dict[str, Any]] = None
    pixel_format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class ContainerBackend(ABC):
    """Abstract base class for container backends"""
    
    @abstractmethod
    def open(self, path: str, mode: str) -> None:
        """Open a container file"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the container"""
        pass
    
    @abstractmethod
    def add_stream(self, metadata: StreamMetadata) -> int:
        """Add a new stream to the container
        
        Returns:
            int: Stream index
        """
        pass
    
    @abstractmethod
    def get_streams(self) -> List[StreamMetadata]:
        """Get list of all streams in the container"""
        pass
    
    @abstractmethod
    def encode_frame(self, frame: Frame, stream_index: int) -> List[bytes]:
        """Encode a frame into packets
        
        Returns:
            List[bytes]: List of encoded packets
        """
        pass
    
    @abstractmethod
    def decode_frame(self, packet: bytes, stream_index: int) -> Frame:
        """Decode a packet into a frame"""
        pass
    
    @abstractmethod
    def mux(self, packet: bytes, stream_index: int) -> None:
        """Write a packet to the container"""
        pass
    
    @abstractmethod
    def demux(self) -> List[tuple[bytes, int]]:
        """Read packets from container
        
        Returns:
            List[tuple[bytes, int]]: List of (packet_data, stream_index) tuples
        """
        pass
    
    @abstractmethod
    def seek(self, timestamp: int, stream_index: int) -> None:
        """Seek to specified timestamp in stream"""
        pass

    # New abstractions for containerization

    @abstractmethod
    def create_stream_with_config(self, config: StreamConfig) -> int:
        """Create a stream with full configuration
        
        Returns:
            int: Stream index
        """
        pass

    @abstractmethod
    def encode_data_to_packets(
        self, 
        data: Any, 
        stream_index: int, 
        timestamp: int,
        codec_config: Any
    ) -> List[PacketInfo]:
        """Encode arbitrary data into packets with timestamp handling
        
        Returns:
            List[PacketInfo]: List of packets ready for muxing
        """
        pass

    @abstractmethod
    def flush_stream(self, stream_index: int) -> List[PacketInfo]:
        """Flush any buffered packets from a stream
        
        Returns:
            List[PacketInfo]: Buffered packets
        """
        pass

    @abstractmethod
    def flush_all_streams(self) -> List[PacketInfo]:
        """Flush all streams and return all buffered packets
        
        Returns:
            List[PacketInfo]: All buffered packets from all streams
        """
        pass
    
    @abstractmethod
    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """Mux a PacketInfo object to the container"""
        pass
    
    @abstractmethod
    def transcode_container(
        self, 
        input_path: str, 
        output_path: str,
        stream_configs: Dict[int, StreamConfig],
        visualization_feature: Optional[str] = None
    ) -> None:
        """Transcode a container from one format/encoding to another
        
        Args:
            input_path: Source container path
            output_path: Destination container path  
            stream_configs: Mapping of stream_index -> new StreamConfig
            visualization_feature: Feature to prioritize in stream ordering
        """
        pass
    
    @abstractmethod
    def create_container_with_new_streams(
        self,
        original_path: str,
        new_path: str, 
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig]
    ) -> Dict[int, int]:
        """Create a new container with existing streams plus new ones
        
        Args:
            original_path: Path to existing container
            new_path: Path for new container
            existing_streams: List of (old_stream_index, config) for existing streams
            new_stream_configs: Configs for new streams to add
            
        Returns:
            Dict[int, int]: Mapping from old stream indices to new stream indices
        """
        pass

    @abstractmethod
    def get_stream_info(self, stream_index: int) -> StreamMetadata:
        """Get metadata for a specific stream"""
        pass

    @abstractmethod 
    def validate_packet(self, packet: Any) -> bool:
        """Check if a packet has valid pts (dts may be optional)"""
        pass

    @abstractmethod
    def extract_packet_info(self, packet: Any) -> PacketInfo:
        """Extract PacketInfo from a backend-specific packet object"""
        pass

    @abstractmethod
    def demux_with_info(self) -> List[PacketInfo]:
        """Demux packets and return as PacketInfo objects
        
        Returns:
            List[PacketInfo]: Packets with full metadata
        """
        pass

    @abstractmethod
    def decode_packet_info(self, packet_info: PacketInfo) -> Frame:
        """Decode a PacketInfo into a Frame"""
        pass

    @abstractmethod
    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Get an iterator for demuxing specific streams
        
        Args:
            stream_indices: List of stream indices to demux
            
        Returns:
            Iterator that yields backend-specific packet objects
        """
        pass

    @abstractmethod
    def seek_container(self, timestamp: int, stream_index: int, any_frame: bool = True) -> None:
        """Seek the container to a specific timestamp
        
        Args:
            timestamp: Target timestamp in milliseconds
            stream_index: Reference stream index for seeking
            any_frame: Whether to seek to any frame or keyframes only
        """
        pass

    @abstractmethod
    def decode_stream_frames(self, stream_index: int, packet_data: bytes = None) -> List[Any]:
        """Decode frames from a stream, optionally with packet data
        
        Args:
            stream_index: Index of the stream to decode from
            packet_data: Optional packet data to decode. If None, flush the decoder.
            
        Returns:
            List of decoded frame objects (backend-specific)
        """
        pass

    @abstractmethod
    def get_stream_metadata(self, stream_index: int) -> Dict[str, str]:
        """Get metadata dictionary for a stream
        
        Args:
            stream_index: Index of the stream
            
        Returns:
            Dictionary of metadata key-value pairs
        """
        pass

    @abstractmethod
    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get the codec name for a stream
        
        Args:
            stream_index: Index of the stream
            
        Returns:
            Codec name string
        """
        pass

    @abstractmethod
    def get_feature_type_from_stream(self, stream_index: int) -> Optional[str]:
        """Get the feature type string from stream metadata
        
        Args:
            stream_index: Index of the stream
            
        Returns:
            Feature type string or None if not found
        """
        pass

    @abstractmethod
    def convert_frame_to_array(self, frame: Any, feature_type: Any, format: str = "rgb24") -> Any:
        """Convert a backend-specific frame to numpy array
        
        Args:
            frame: Backend-specific frame object
            feature_type: FeatureType object for reshaping
            format: Pixel format for conversion
            
        Returns:
            Numpy array or processed data
        """
        pass

    @abstractmethod
    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if a stream exists for a given feature name
        
        Args:
            feature_name: Name of the feature to search for
            
        Returns:
            Stream index if found, None otherwise
        """
        pass

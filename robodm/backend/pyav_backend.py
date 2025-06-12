from __future__ import annotations

"""PyAV-backed implementation of the ContainerBackend interface.

This module converts the abstract operations defined in
:`robodm.backend.base.ContainerBackend` into concrete calls against the
`PyAV` API so that the rest of the codebase can remain backend-agnostic.

The guiding principle is **minimum interference** with the existing logic in
`robodm.trajectory.Trajectory`: wherever that class already manipulates
`PyAV` primitives directly, this backend returns or accepts those same
objects so we do **not** have to rewrite the fragile frame-handling code.
"""

import os
import pickle
import logging
from fractions import Fraction
from typing import Any, Dict, List, Tuple, Optional

import av
import numpy as np

from .base import ContainerBackend, Frame, StreamMetadata, PacketInfo, StreamConfig

logger = logging.getLogger(__name__)


class PyAVBackend(ContainerBackend):
    """ContainerBackend implementation that relies on the PyAV library.

    Notes
    -----
    * The backend keeps a reference to the underlying :class:`av.container.InputContainer`
      or :class:`av.container.OutputContainer` in ``self.container`` so that legacy
      code can keep using `container.mux(...)`, `container.streams`, etc.
    * All timestamps are interpreted in **milliseconds** – this mirrors the rest
      of the codebase where the time-base is hard-coded to ``Fraction(1, 1000)``.
    """

    DEFAULT_FORMAT: str = "matroska"

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def __init__(self, container_format: str | None = None) -> None:
        self.container_format: str = container_format or self.DEFAULT_FORMAT
        self.container: av.container.Container | None = None
        # Map index -> av.Stream for quick lookup
        self._idx_to_stream: Dict[int, av.stream.Stream] = {}

    # ------------------------------------------------------------------
    # API implementation
    # ------------------------------------------------------------------
    def open(self, path: str, mode: str) -> None:  # noqa: D401  (docstring inherited)
        if mode not in {"r", "w"}:
            raise ValueError("mode must be 'r' or 'w'")
        self.container = av.open(path, mode=mode, format=self.container_format)
        # Populate mapping for existing streams (in read mode).
        if mode == "r":
            self._idx_to_stream = {
                s.index: s for s in self.container.streams  # type: ignore[index]
            }

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None
            self._idx_to_stream.clear()

    def add_stream(self, metadata: StreamMetadata) -> int:
        if self.container is None:
            raise RuntimeError("Container not opened")
        stream = self.container.add_stream(metadata.encoding)

        # Set metadata on stream
        stream.metadata["FEATURE_NAME"] = metadata.feature_name
        stream.metadata["FEATURE_TYPE"] = metadata.feature_type

        # Time-base
        num, den = metadata.time_base
        stream.time_base = Fraction(num, den)

        # Additional metadata
        if metadata.additional_metadata:
            for k, v in metadata.additional_metadata.items():
                stream.metadata[k] = v

        # Save mapping and return index
        self._idx_to_stream[stream.index] = stream
        return stream.index

    def get_streams(self) -> List[StreamMetadata]:
        out: List[StreamMetadata] = []
        for idx, stream in self._idx_to_stream.items():
            fn = stream.metadata.get("FEATURE_NAME", f"stream_{idx}")
            ft = stream.metadata.get("FEATURE_TYPE", "unknown")
            enc = stream.codec_context.codec.name
            tb = (stream.time_base.numerator, stream.time_base.denominator)
            out.append(
                StreamMetadata(
                    feature_name=fn,
                    feature_type=ft,
                    encoding=enc,
                    time_base=tb,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Encoding / decoding helpers
    # ------------------------------------------------------------------
    def encode_frame(self, frame: Frame, stream_index: int) -> List[bytes]:
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        codec_name = stream.codec_context.codec.name

        packets: List[bytes] = []

        # Video path (numpy ndarray → VideoFrame)
        if isinstance(frame.data, np.ndarray) and codec_name != "rawvideo":
            # We always assume RGB24 input here – higher-level code is
            # responsible for ensuring shape / dtype compatibility.
            vframe = av.VideoFrame.from_ndarray(frame.data, format="rgb24")
            # PyAV requires re-setting pts/dts on the VideoFrame
            vframe.pts = frame.pts
            vframe.dts = frame.dts
            vframe.time_base = Fraction(*frame.time_base)

            for pkt in stream.encode(vframe):  # type: ignore[attr-defined]
                packets.append(bytes(pkt))
        else:
            # Raw path (typically pickled data)
            pkt = av.Packet(frame.data if isinstance(frame.data, (bytes, bytearray)) else bytes(frame.data))
            pkt.pts = frame.pts
            pkt.dts = frame.dts
            pkt.time_base = Fraction(*frame.time_base)
            pkt.stream = stream
            packets.append(bytes(pkt))

        return packets

    def decode_frame(self, packet: bytes, stream_index: int) -> Frame:
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        stream = self._idx_to_stream[stream_index]
        pkt = av.Packet(packet)
        pkt.stream = stream

        # Decode – may return 0-N frames; we only care about the first one for now
        frames = pkt.decode()
        if frames:
            frm = frames[0]
            arr = frm.to_ndarray(format="rgb24")
            return Frame(
                data=arr,
                pts=int(frm.pts or 0),
                dts=int(frm.dts or 0),
                time_base=(stream.time_base.numerator, stream.time_base.denominator),
                stream_index=stream_index,
                is_keyframe=bool(frm.key_frame),
            )
        # Fallback: raw packet (e.g. pickled data)
        return Frame(
            data=packet,
            pts=int(pkt.pts or 0),
            dts=int(pkt.dts or 0),
            time_base=(stream.time_base.numerator, stream.time_base.denominator),
            stream_index=stream_index,
            is_keyframe=False,
        )

    # ------------------------------------------------------------------
    # Mux / demux / seek wrappers
    # ------------------------------------------------------------------
    def mux(self, packet: bytes, stream_index: int) -> None:
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")

        pkt = av.Packet(packet)
        pkt.stream = self._idx_to_stream[stream_index]
        self.container.mux(pkt)

    def demux(self) -> List[Tuple[bytes, int]]:
        if self.container is None:
            raise RuntimeError("Container not opened")
        out: List[Tuple[bytes, int]] = []
        for pkt in self.container.demux(self.container.streams):  # type: ignore[arg-type]
            out.append((bytes(pkt), pkt.stream.index))
        return out

    def seek(self, timestamp: int, stream_index: int) -> None:
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        self.container.seek(timestamp, stream=self._idx_to_stream[stream_index], any_frame=True)

    # ------------------------------------------------------------------
    # New containerization abstractions
    # ------------------------------------------------------------------
    
    def create_stream_with_config(self, config: StreamConfig) -> int:
        """Create a stream with full configuration"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        
        stream = self.container.add_stream(config.encoding)
        
        # Configure stream for video codecs
        if config.encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            if config.width and config.height:
                stream.width = config.width
                stream.height = config.height
            elif hasattr(config.feature_type, 'shape') and config.feature_type.shape:
                shape = config.feature_type.shape
                if len(shape) >= 2:
                    stream.width = shape[1]
                    stream.height = shape[0]

            if config.pixel_format:
                stream.pix_fmt = config.pixel_format

            if config.codec_options:
                stream.codec_context.options = config.codec_options

        # Metadata and time-base
        stream.metadata["FEATURE_NAME"] = config.feature_name
        stream.metadata["FEATURE_TYPE"] = str(config.feature_type)
        stream.time_base = Fraction(1, 1000)

        self._idx_to_stream[stream.index] = stream
        return stream.index

    def encode_data_to_packets(
        self, 
        data: Any, 
        stream_index: int, 
        timestamp: int,
        codec_config: Any
    ) -> List[PacketInfo]:
        """Encode arbitrary data into packets with timestamp handling"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        encoding = stream.codec_context.codec.name
        
        packets: List[PacketInfo] = []
        
        # Determine if this should be encoded as video or raw
        if (encoding in {"ffv1", "libaom-av1", "libx264", "libx265"} and 
            isinstance(data, np.ndarray) and len(data.shape) >= 2):
            
            # Create video frame
            frame = self._create_frame(data, stream)
            frame.time_base = stream.time_base
            frame.pts = timestamp
            frame.dts = timestamp
            
            # Encode to packets
            for pkt in stream.encode(frame):  # type: ignore[attr-defined]
                packets.append(PacketInfo(
                    data=bytes(pkt),
                    pts=pkt.pts,
                    dts=pkt.dts,
                    stream_index=stream_index,
                    time_base=(stream.time_base.numerator, stream.time_base.denominator),
                    is_keyframe=bool(pkt.is_keyframe) if hasattr(pkt, 'is_keyframe') else False
                ))
        else:
            # Raw/pickled data path
            if isinstance(data, np.ndarray):
                payload = pickle.dumps(data)
            else:
                payload = pickle.dumps(data)

            packets.append(PacketInfo(
                data=payload,
                pts=timestamp,
                dts=timestamp,
                stream_index=stream_index,
                time_base=(stream.time_base.numerator, stream.time_base.denominator),
                is_keyframe=True
            ))
        
        return packets

    def flush_stream(self, stream_index: int) -> List[PacketInfo]:
        """Flush any buffered packets from a stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        packets: List[PacketInfo] = []
        
        try:
            # Flush the encoder
            for pkt in stream.encode(None):  # type: ignore[attr-defined]
                packets.append(PacketInfo(
                    data=bytes(pkt),
                    pts=pkt.pts,
                    dts=pkt.dts,
                    stream_index=stream_index,
                    time_base=(stream.time_base.numerator, stream.time_base.denominator),
                    is_keyframe=bool(pkt.is_keyframe) if hasattr(pkt, 'is_keyframe') else False
                ))
        except av.error.EOFError:
            # Expected when encoder is fully flushed
            pass
        except Exception as e:
            logger.error(f"Error flushing stream {stream_index}: {e}")
        
        return packets

    def flush_all_streams(self) -> List[PacketInfo]:
        """Flush all streams and return all buffered packets"""
        packets: List[PacketInfo] = []
        for stream_index in self._idx_to_stream:
            packets.extend(self.flush_stream(stream_index))
        return packets
    
    def mux_packet_info(self, packet_info: PacketInfo) -> None:
        """Mux a PacketInfo object to the container"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        if packet_info.stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {packet_info.stream_index}")

        pkt = av.Packet(packet_info.data)
        pkt.pts = packet_info.pts
        pkt.dts = packet_info.dts
        pkt.time_base = Fraction(*packet_info.time_base)
        pkt.stream = self._idx_to_stream[packet_info.stream_index]
        
        self.container.mux(pkt)
    
    def transcode_container(
        self, 
        input_path: str, 
        output_path: str,
        stream_configs: Dict[int, StreamConfig],
        visualization_feature: Optional[str] = None
    ) -> None:
        """Transcode a container from one format/encoding to another"""
        
        # Open input container
        input_container = av.open(input_path, mode="r", format=self.container_format)
        input_streams = list(input_container.streams)
        
        # Create output container
        output_container = av.open(output_path, mode="w", format=self.container_format)
        
        # Sort streams to prioritize visualization feature
        def get_stream_priority(stream):
            feature_name = stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                return (3, stream.index)
            
            # Highest priority: specified visualization_feature
            if visualization_feature and feature_name == visualization_feature:
                return (0, stream.index)
            
            # Second priority: streams that will become video-encoded
            if stream.index in stream_configs:
                config = stream_configs[stream.index]
                if config.encoding != "rawvideo":
                    return (1, stream.index)
            
            # Third priority: everything else
            return (2, stream.index)
        
        sorted_streams = sorted(input_streams, key=get_stream_priority)
        
        # Create output streams
        stream_mapping: Dict[int, int] = {}
        for input_stream in sorted_streams:
            feature_name = input_stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                continue
            
            if input_stream.index in stream_configs:
                config = stream_configs[input_stream.index]
                output_stream_idx = self._create_output_stream(output_container, config)
            else:
                # Copy existing stream configuration
                config = StreamConfig(
                    feature_name=feature_name,
                    feature_type=input_stream.metadata.get("FEATURE_TYPE", "unknown"),
                    encoding=input_stream.codec_context.codec.name
                )
                output_stream_idx = self._create_output_stream(output_container, config)
            
            stream_mapping[input_stream.index] = output_stream_idx
        
        # Process packets
        packets_muxed = 0
        for packet in input_container.demux(input_streams):
            if not self.validate_packet(packet):
                logger.debug(f"Skipping invalid packet: {packet}")
                continue
            
            if packet.stream.index not in stream_mapping:
                continue
                
            output_stream_idx = stream_mapping[packet.stream.index]
            output_stream = output_container.streams[output_stream_idx]
            
            # Check if we need to transcode
            original_encoding = packet.stream.codec_context.codec.name
            target_config = stream_configs.get(packet.stream.index)
            
            if (original_encoding == "rawvideo" and target_config and 
                target_config.encoding != "rawvideo"):
                # Transcode from pickled to video
                data = pickle.loads(bytes(packet))
                frame = self._create_frame(data, output_stream)
                frame.time_base = output_stream.time_base  
                frame.pts = packet.pts
                frame.dts = packet.dts
                
                for new_packet in output_stream.encode(frame):  # type: ignore[attr-defined]
                    output_container.mux(new_packet)
                    packets_muxed += 1
            else:
                # Direct remux
                packet.stream = output_stream
                output_container.mux(packet)
                packets_muxed += 1
        
        # Flush all output streams
        for stream in output_container.streams:
            try:
                for packet in stream.encode(None):  # type: ignore[attr-defined]
                    output_container.mux(packet)
                    packets_muxed += 1
            except Exception as e:
                logger.error(f"Error flushing output stream {stream}: {e}")
        
        logger.debug(f"Transcoding complete: {packets_muxed} packets muxed")
        
        input_container.close()
        output_container.close()

    def create_container_with_new_streams(
        self,
        original_path: str,
        new_path: str, 
        existing_streams: List[Tuple[int, StreamConfig]],
        new_stream_configs: List[StreamConfig]
    ) -> Dict[int, int]:
        """Create a new container with existing streams plus new ones"""
        
        # Open original container
        original_container = av.open(original_path, mode="r", format=self.container_format)
        original_stream_objects = list(original_container.streams)
        
        # Create new container  
        new_container = av.open(new_path, mode="w", format=self.container_format)
        
        stream_mapping: Dict[int, int] = {}
        
        # Add existing streams
        for old_idx, config in existing_streams:
            new_idx = self._create_output_stream(new_container, config)
            stream_mapping[old_idx] = new_idx
        
        # Add new streams
        for config in new_stream_configs:
            new_idx = self._create_output_stream(new_container, config)
            # New streams don't have an old index to map from
        
        # Copy existing packets
        for packet in original_container.demux(original_stream_objects):
            if not self.validate_packet(packet):
                continue
                
            if packet.stream.index in stream_mapping:
                new_stream_idx = stream_mapping[packet.stream.index]
                packet.stream = new_container.streams[new_stream_idx]
                new_container.mux(packet)
        
        original_container.close()
        
        # Keep new container open and update our state
        if self.container is not None:
            self.container.close()
        self.container = new_container
        self._idx_to_stream = {s.index: s for s in new_container.streams}
        
        return stream_mapping

    def get_stream_info(self, stream_index: int) -> StreamMetadata:
        """Get metadata for a specific stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        feature_name = stream.metadata.get("FEATURE_NAME", f"stream_{stream_index}")
        feature_type = stream.metadata.get("FEATURE_TYPE", "unknown")
        encoding = stream.codec_context.codec.name
        time_base = (stream.time_base.numerator, stream.time_base.denominator)
        
        return StreamMetadata(
            feature_name=feature_name,
            feature_type=feature_type,
            encoding=encoding,
            time_base=time_base
        )

    def validate_packet(self, packet: Any) -> bool:
        """Check if a packet has valid pts/dts"""
        # Only check pts like the original code - some packets may not have dts
        return packet.pts is not None

    def extract_packet_info(self, packet: Any) -> PacketInfo:
        """Extract PacketInfo from a PyAV packet object"""
        return PacketInfo(
            data=bytes(packet),
            pts=packet.pts,
            dts=packet.dts,
            stream_index=packet.stream.index,
            time_base=(packet.time_base.numerator, packet.time_base.denominator),
            is_keyframe=bool(packet.is_keyframe) if hasattr(packet, 'is_keyframe') else False
        )

    def demux_with_info(self) -> List[PacketInfo]:
        """Demux packets and return as PacketInfo objects"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        
        packets: List[PacketInfo] = []
        for pkt in self.container.demux(self.container.streams):  # type: ignore[arg-type]
            packets.append(self.extract_packet_info(pkt))
        return packets

    def decode_packet_info(self, packet_info: PacketInfo) -> Frame:
        """Decode a PacketInfo into a Frame"""
        return self.decode_frame(packet_info.data, packet_info.stream_index)

    def demux_streams(self, stream_indices: List[int]) -> Any:
        """Get an iterator for demuxing specific streams"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        
        # Get the actual stream objects for the given indices
        streams = [self._idx_to_stream[idx] for idx in stream_indices if idx in self._idx_to_stream]
        return self.container.demux(streams)

    def seek_container(self, timestamp: int, stream_index: int, any_frame: bool = True) -> None:
        """Seek the container to a specific timestamp"""
        if self.container is None:
            raise RuntimeError("Container not opened")
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        self.container.seek(timestamp, stream=stream, any_frame=any_frame)

    def decode_stream_frames(self, stream_index: int, packet_data: bytes = None) -> List[Any]:
        """Decode frames from a stream, optionally with packet data"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        
        if packet_data is None:
            # Flush decoder
            return list(stream.decode(None))
        else:
            # Decode specific packet
            pkt = av.Packet(packet_data)
            pkt.stream = stream
            return list(pkt.decode())

    def get_stream_metadata(self, stream_index: int) -> Dict[str, str]:
        """Get metadata dictionary for a stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        return dict(stream.metadata)

    def get_stream_codec_name(self, stream_index: int) -> str:
        """Get the codec name for a stream"""
        if stream_index not in self._idx_to_stream:
            raise ValueError(f"No stream with index {stream_index}")
        
        stream = self._idx_to_stream[stream_index]
        return stream.codec_context.codec.name

    def get_feature_type_from_stream(self, stream_index: int) -> Optional[str]:
        """Get the feature type string from stream metadata"""
        if stream_index not in self._idx_to_stream:
            return None
        
        stream = self._idx_to_stream[stream_index]
        return stream.metadata.get("FEATURE_TYPE")

    def convert_frame_to_array(self, frame: Any, feature_type: Any, format: str = "rgb24") -> Any:
        """Convert a backend-specific frame to numpy array"""
        import pickle
        
        # Handle pickled data (rawvideo packets)
        if isinstance(frame, bytes):
            return pickle.loads(frame)
        
        # Handle PyAV video frames
        if hasattr(frame, 'to_ndarray'):
            # Check if this is RGB data that should be decoded as RGB24
            if (hasattr(feature_type, 'shape') and feature_type.shape and 
                len(feature_type.shape) == 3 and feature_type.shape[2] == 3):
                arr = frame.to_ndarray(format=format)
            else:
                # For non-RGB data, this might be an issue but handle gracefully
                arr = frame.to_ndarray(format=format)
            
            # Reshape if needed
            if hasattr(feature_type, 'shape') and feature_type.shape:
                arr = arr.reshape(feature_type.shape)
            
            return arr
        
        # Fallback - return as is
        return frame

    def stream_exists_by_feature(self, feature_name: str) -> Optional[int]:
        """Check if a stream exists for a given feature name"""
        for stream_idx, stream in self._idx_to_stream.items():
            if stream.metadata.get("FEATURE_NAME") == feature_name:
                return stream_idx
        return None

    # ------------------------------------------------------------------
    # High-level helpers that map directly from Trajectory logic
    # ------------------------------------------------------------------
    def add_stream_for_feature(
        self,
        feature_name: str,
        feature_type: "FeatureType",
        codec_config: "CodecConfig",
        encoding: str | None = None,
    ) -> "av.stream.Stream":
        """Create a new stream inside the currently opened container.

        This mirrors the logic previously found in
        ``Trajectory._add_stream_to_container`` so that that method can now be
        reduced to a thin wrapper that delegates to this backend.
        """

        if self.container is None:
            raise RuntimeError("Container not opened")

        # Determine encoding if not explicitly provided.
        enc = encoding or codec_config.get_codec_for_feature(feature_type)

        stream = self.container.add_stream(enc)

        # Configure stream for video codecs
        if enc in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            shape = feature_type.shape
            if shape is not None and len(shape) >= 2:
                stream.width = shape[1]
                stream.height = shape[0]

            pixel_fmt = codec_config.get_pixel_format(enc, feature_type)
            if pixel_fmt:
                stream.pix_fmt = pixel_fmt

            codec_opts = codec_config.get_codec_options(enc)
            if codec_opts:
                stream.codec_context.options = codec_opts

        # Metadata and time-base
        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)

        self._idx_to_stream[stream.index] = stream
        return stream

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    def _create_output_stream(self, container: av.container.OutputContainer, config: StreamConfig) -> int:
        """Helper to create a stream in an output container"""
        stream = container.add_stream(config.encoding)
        
        # Configure video codec settings
        if config.encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            if config.width and config.height:
                stream.width = config.width
                stream.height = config.height
            elif hasattr(config.feature_type, 'shape'):
                shape = getattr(config.feature_type, 'shape', None)
                if shape and len(shape) >= 2:
                    stream.width = shape[1]
                    stream.height = shape[0]

            if config.pixel_format:
                stream.pix_fmt = config.pixel_format

            if config.codec_options:
                stream.codec_context.options = config.codec_options

        # Set metadata
        stream.metadata["FEATURE_NAME"] = config.feature_name
        stream.metadata["FEATURE_TYPE"] = str(config.feature_type)
        stream.time_base = Fraction(1, 1000)
        
        return stream.index

    # The following helpers replicate the fragile image handling logic that
    # previously lived in Trajectory.

    def _create_frame(self, image_array, stream):
        import numpy as _np

        image_array = _np.array(image_array)
        encoding = stream.codec_context.codec.name

        # Convert to uint8 if needed
        if image_array.dtype == _np.float32:
            image_array = _np.clip(image_array * 255, 0, 255).astype(_np.uint8)
        elif image_array.dtype != _np.uint8:
            if _np.issubdtype(image_array.dtype, _np.integer):
                image_array = _np.clip(image_array, 0, 255).astype(_np.uint8)
            else:
                image_array = _np.clip(image_array * 255, 0, 255).astype(_np.uint8)

        # Only handle RGB images (HxWx3)
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(
                "Video codecs only support RGB images with shape (H, W, 3). "
                f"Got shape {image_array.shape}."
            )

        # Create RGB frame and convert to YUV420p when required.
        if encoding in {"libaom-av1", "ffv1", "libx264", "libx265"}:
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")
            frame = frame.reformat(format="yuv420p")
        else:
            frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")

        return frame

    def _create_frame_depth(self, image_array, stream):
        import numpy as _np

        image_array = _np.array(image_array)

        if image_array.dtype == _np.float32:
            image_array = (image_array * 255).astype(_np.uint8)

        if len(image_array.shape) == 3:
            if image_array.shape[2] == 3:
                image_array = _np.mean(image_array, axis=2).astype(_np.uint8)
            else:
                image_array = image_array[:, :, 0]

        frame = av.VideoFrame.from_ndarray(image_array, format="gray")
        frame.time_base = stream.time_base
        return frame

    def encode_data(
        self,
        data: Any,
        stream: "av.stream.Stream",
        timestamp: int,
        codec_config: "CodecConfig",
    ) -> List["av.packet.Packet"]:
        """Encode arbitrary *data* into packets for *stream* following the
        original logic of Trajectory._encode_frame.
        """

        from robodm.feature import FeatureType  # local import to avoid cycles

        encoding = stream.codec_context.codec.name
        feature_type = FeatureType.from_data(data)

        packets: List[av.Packet]

        if (
            encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}
            and feature_type.shape is not None
            and len(feature_type.shape) >= 2
        ):
            frame = self._create_frame(data, stream)
            frame.time_base = stream.time_base
            frame.pts = timestamp
            frame.dts = timestamp
            packets = list(stream.encode(frame))  # type: ignore[attr-defined]
        else:
            # Fallback to pickled rawvideo path
            import pickle, numpy as _np

            if isinstance(data, _np.ndarray):
                payload = pickle.dumps(data)
            else:
                payload = pickle.dumps(data)

            pkt = av.Packet(payload)
            pkt.pts = timestamp
            pkt.dts = timestamp
            pkt.time_base = stream.time_base
            pkt.stream = stream
            packets = [pkt]

        return packets 
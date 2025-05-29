from fractions import Fraction
import logging
import time
from typing import Any, Dict, List, Optional, Text
import av
import numpy as np
import os
from fog_x import FeatureType
import pickle
from fog_x.utils import recursively_read_hdf5_group
import h5py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys

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


class StreamInfo:
    def __init__(self, feature_name, feature_type, encoding):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.encoding = encoding

    def __str__(self):
        return f"StreamInfo({self.feature_name}, {self.feature_type}, {self.encoding})"

    def __repr__(self):
        return self.__str__()


class Trajectory:
    def __init__(
        self,
        path: Text,
        mode="r",
        cache_dir: Optional[Text] = "/tmp/fog_x/cache/",
        lossy_compression: bool = True,
        feature_name_separator: Text = "/",
        filesystem: Optional[Any] = None,
        time_provider: Optional[Any] = None,
    ) -> None:
        """
        Args:
            path (Text): path to the trajectory file
            mode (Text, optional):  mode of the file, "r" for read and "w" for write
            num_pre_initialized_h264_streams (int, optional):
                Number of pre-initialized H.264 video streams to use when adding new features.
                we pre initialize a configurable number of H.264 video streams to avoid the overhead of creating new streams for each feature.
                otherwise we need to remux everytime
            . Defaults to 5.
            feature_name_separator (Text, optional):
                Delimiter to separate feature names in the container file.
                Defaults to "/".
            filesystem: Optional filesystem interface for dependency injection
            time_provider: Optional time provider interface for dependency injection
        """
        self.path = path
        self.feature_name_separator = feature_name_separator
        
        # Dependency injection - set early so they're available during init
        self._filesystem = filesystem
        self._time_provider = time_provider
        
        # self.cache_file_name = "/tmp/fog_" + os.path.basename(self.path) + ".cache"
        # use hex hash of the path for the cache file name
        if not self._exists(cache_dir):
            self._makedirs(cache_dir, exist_ok=True)
        hex_hash = hex(abs(hash(self.path)))[2:]
        self.cache_file_name = cache_dir + hex_hash + ".cache"
        # self.cache_file_name = cache_dir + os.path.basename(self.path) + ".cache"
        self.feature_name_to_stream = {}  # feature_name: stream
        self.feature_name_to_feature_type = {}  # feature_name: feature_type
        self.trajectory_data = None  # trajectory_data
        self.start_time = self._time()
        self.mode = mode
        self.stream_id_to_info = {}  # stream_id: StreamInfo
        self.is_closed = False
        self.lossy_compression = lossy_compression
        self.pending_write_tasks = []  # List to keep track of pending write tasks
        # self.cache_write_lock = asyncio.Lock()
        # self.cache_write_task = None
        # self.executor = ThreadPoolExecutor(max_workers=1)

        # check if the path exists
        # if not, create a new file and start data collection
        if self.mode == "w":
            if not self._exists(self.path):
                self._makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                self.container_file = av.open(self.path, mode="w", format="matroska")
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
        current_time = (self._time() - self.start_time) * 1000
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

        return self.trajectory_data[key]

    def close(self, compact=True):
        """
        close the container file

        args:
        compact: re-read from the cache to encode pickled data to images
        """
        logger.debug(f"Closing trajectory, is_closed={self.is_closed}, compact={compact}")
        
        if self.is_closed:
            raise ValueError("The container file is already closed")
        
        logger.debug(f"Container has {len(self.container_file.streams)} streams")
        
        # Check if there are any streams with data
        has_data = len(self.container_file.streams) > 0
        
        try:
            ts = self._get_current_timestamp()
            logger.debug(f"Final timestamp: {ts}")
            
            for i, stream in enumerate(self.container_file.streams):
                logger.debug(f"Flushing stream {i}: {stream}")
                try:
                    packets = stream.encode(None)
                    logger.debug(f"Stream {i} flush returned {len(packets)} packets")
                    for j, packet in enumerate(packets):
                        packet.pts = ts
                        packet.dts = ts
                        self.container_file.mux(packet)
                        logger.debug(f"Muxed flush packet {j} from stream {i}")
                except Exception as e:
                    logger.error(f"Error flushing stream {stream}: {e}")
            logger.debug("Flushing the container file")
        except av.error.EOFError:
            logger.debug("Got EOFError during flush (expected)")
            pass  # This exception is expected and means the encoder is fully flushed

        logger.debug("Closing container file")
        self.container_file.close()
        
        # Only attempt transcoding if file exists, has content, and compact is requested
        if compact and has_data and self._exists(self.path) and os.path.getsize(self.path) > 0:
            logger.debug("Starting transcoding of pickled images")
            # After closing, re-read from the cache to encode pickled data to images
            self._transcode_pickled_images(ending_timestamp=ts)
        else:
            logger.debug(f"Skipping transcoding: compact={compact}, has_data={has_data}, file_exists={self._exists(self.path)}, file_size={os.path.getsize(self.path) if self._exists(self.path) else 0}")
        
        self.trajectory_data = None
        self.container_file = None
        self.is_closed = True
        logger.debug("Trajectory closed successfully")

    def load(self, save_to_cache=True, return_type="numpy"):
        """
        Load the trajectory data.

        Args:
            mode (str): "cache" to use cached data if available, "no_cache" to always load from container.
            return_h5 (bool): If True, return h5py.File object instead of numpy arrays.

        Returns:
            dict: A dictionary of numpy arrays if return_h5 is False, otherwise an h5py.File object.
        """

        # uncomment the following line to use async
        # return asyncio.get_event_loop().run_until_complete(
        #     self.load_async(save_to_cache=save_to_cache, return_h5=return_h5)
        # )
        # async def load_async(self, save_to_cache=True, return_h5=False):
        np_cache = None
        if not self._exists(self.cache_file_name):
            logger.debug(f"Loading the container file {self.path}, saving to cache {self.cache_file_name}")
            np_cache = self._load_from_container()
            if save_to_cache:
                # await self._async_write_to_cache(np_cache)
                try:
                    self._write_to_cache(np_cache)
                except Exception as e:
                    logger.error(f"Error writing to cache file {self.cache_file_name}: {e}")
                    return np_cache
        
        if return_type =="hdf5":
            return h5py.File(self.cache_file_name, "r")
        elif return_type == "numpy":
            if not np_cache:
                try:
                    with h5py.File(self.cache_file_name, "r") as h5_cache:
                        np_cache = recursively_read_hdf5_group(h5_cache)
                except Exception as e:
                    logger.error(f"Error loading cache file {self.cache_file_name}: {e}, reading from container")
                    np_cache = self._load_from_container()
            return np_cache
        elif return_type == "cache_name":
            return self.cache_file_name
        elif return_type == "container":
            return self.path
        elif return_type == "tensor":
            import tensorflow as tf
            def _convert_h5_cache_to_tensor(h5_cache):
                output_tf_traj = {}
                for key in h5_cache:
                    # hierarhical 
                    if type(h5_cache[key]) == h5py._hl.group.Group:
                        for sub_key in h5_cache[key]:
                            if key not in output_tf_traj:
                                output_tf_traj[key] = {}
                            output_tf_traj[key][sub_key] = tf.convert_to_tensor(h5_cache[key][sub_key])
                    elif type(h5_cache[key]) == h5py._hl.dataset.Dataset:
                        output_tf_traj[key] = tf.convert_to_tensor(h5_cache[key])
                return output_tf_traj
            with h5py.File(self.cache_file_name, 'r') as h5_cache:
                # Step 2: Access the dataset within the file
                # Assume the dataset is named 'dataset_name'
                output_traj = _convert_h5_cache_to_tensor(h5_cache)
            return output_traj
        else:
            raise ValueError(f"Invalid return_type {return_type}")
            
            

    def init_feature_streams(self, feature_spec: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_spec.items():
            encoding = self._get_encoding_of_feature(None, feature_type)
            self.feature_name_to_stream[feature] = self._add_stream_to_container(
                self.container_file, feature, encoding, feature_type
            )

    def add(
        self,
        feature: str,
        data: Any,
        timestamp: Optional[int] = None,
    ) -> None:
        """
        add one value to container file

        Args:
            feature (str): name of the feature
            value (Any): value associated with the feature; except dictionary
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.

        Examples:
            >>> trajectory.add('feature1', 'image1.jpg')

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
        logger.debug(f"Adding feature: {feature}, data shape: {getattr(data, 'shape', 'N/A')}")

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

        # get the timestamp
        if timestamp is None:
            timestamp = self._get_current_timestamp()

        logger.debug(f"Encoding frame with timestamp: {timestamp}")
        # encode the frame
        packets = self._encode_frame(data, stream, timestamp)
        logger.debug(f"Generated {len(packets)} packets")

        # write the packet to the container
        for i, packet in enumerate(packets):
            logger.debug(f"Muxing packet {i}: {packet}")
            self.container_file.mux(packet)
            logger.debug(f"Successfully muxed packet {i}")

    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
    ) -> None:
        """
        add one value to container file
        data might be nested dictionary of values for each feature

        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): nanoseconds since the Epoch.
                If not provided, the current time is used.
                assume the timestamp is same for all the features within the dictionary

        Examples:
            >>> trajectory.add_by_dict({'feature1': 'image1.jpg'})

        Logic:
        - check the data see if it is a dictionary
        - if dictionary, need to flatten it and add each feature separately
        """
        if type(data) != dict:
            raise ValueError("Use add for non-dictionary data, type is ", type(data))

        _flatten_dict_data = _flatten_dict(data, sep=self.feature_name_separator)
        timestamp = self._get_current_timestamp() if timestamp is None else timestamp
        for feature, value in _flatten_dict_data.items():
            self.add(feature, value, timestamp)

    @classmethod
    def from_list_of_dicts(cls, data: List[Dict[str, Any]], path: Text, lossy_compression: bool = True) -> "Trajectory":
        """
        Create a Trajectory object from a list of dictionaries.

        args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file

        Example:
        original_trajectory = [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"},
        ]

        trajectory = Trajectory.from_list_of_dicts(original_trajectory, path="/tmp/fog_x/output.vla")
        """
        traj = cls(path, mode="w", lossy_compression=lossy_compression)
        logger.info(f"Creating a new trajectory file at {path} with {len(data)} steps")
        for step in data:
            traj.add_by_dict(step)
        traj.close()
        return traj

    @classmethod
    def from_dict_of_lists(
        cls, data: Dict[str, List[Any]], path: Text, feature_name_separator: Text = "/", lossy_compression: bool = True
    ) -> "Trajectory":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file

        Returns:
            Trajectory: _description_

        Example:
        original_trajectory = {
            "feature1": ["value1", "value3"],
            "feature2": ["value2", "value4"],
        }

        trajectory = Trajectory.from_dict_of_lists(original_trajectory, path="/tmp/fog_x/output.vla")
        """
        traj = cls(path, feature_name_separator=feature_name_separator, mode="w", lossy_compression = lossy_compression)
        # flatten the data such that all data starts and put feature name with separator
        _flatten_dict_data = _flatten_dict(data, sep=traj.feature_name_separator)

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

    def _load_from_cache(self):
        """
        load the cached file with entire vla trajctory
        """
        h5_cache = h5py.File(self.cache_file_name, "r")
        return h5_cache

    def _load_from_container(self):
        """
        Load the container file with the entire VLA trajectory using multi-processing for image streams.
        
        args:
            save_to_cache: save the decoded data to the cache file
        
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
        np_cache_lists = {}
        feature_name_to_stream = {}

        # Initialize lists for each feature
        for stream in streams:
            feature_name = stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                continue
            feature_type = FeatureType.from_str(stream.metadata.get("FEATURE_TYPE"))
            feature_name_to_stream[feature_name] = stream
            self.feature_name_to_feature_type[feature_name] = feature_type

            logger.debug(f"Initializing list for {feature_name} with feature_type {feature_type}")
            np_cache_lists[feature_name] = []

        # Decode the frames and store them in the lists
        for packet in container.demux(list(streams)):
            feature_name = packet.stream.metadata.get("FEATURE_NAME")
            if feature_name is None:
                logger.debug(f"Skipping stream without FEATURE_NAME: {packet.stream}")
                continue
            feature_type = FeatureType.from_str(packet.stream.metadata.get("FEATURE_TYPE"))

            logger.debug(f"Decoding {feature_name} with time {packet.dts}")

            feature_codec = packet.stream.codec_context.codec.name
            if feature_codec == "rawvideo":
                packet_in_bytes = bytes(packet)
                if packet_in_bytes:
                    # Decode the packet
                    data = pickle.loads(packet_in_bytes)
                    np_cache_lists[feature_name].append(data)
                else:
                    logger.debug(f"Skipping empty packet: {packet} for {feature_name}")
            else:
                frames = packet.decode()
                for frame in frames:
                    if feature_type.dtype == "float32":
                        data = frame.to_ndarray(format="gray").reshape(feature_type.shape)
                    else:
                        data = frame.to_ndarray(format="rgb24").reshape(feature_type.shape)
                    np_cache_lists[feature_name].append(data)

        container.close()

        # Convert lists to numpy arrays
        np_cache = {}
        for feature_name, data_list in np_cache_lists.items():
            logger.debug(f"Converting {feature_name} list of length {len(data_list)} to numpy array")
            if not data_list:
                logger.debug(f"Warning: {feature_name} has no data!")
                continue
                
            feature_type = self.feature_name_to_feature_type[feature_name]
            
            if feature_type.dtype == "string":
                np_cache[feature_name] = np.array(data_list, dtype=object)
            else:
                # Convert list to numpy array
                np_cache[feature_name] = np.array(data_list, dtype=feature_type.dtype)

        return np_cache

    # async def _async_write_to_cache(self, np_cache):
    #     async with self.cache_write_lock:
    #         await asyncio.get_event_loop().run_in_executor(
    #             self.executor,
    #             self._write_to_cache,
    #             np_cache
    #         )

    def _write_to_cache(self, np_cache):
        try:
            h5_cache = h5py.File(self.cache_file_name, "w")
        except Exception as e:
            logger.error(f"Error creating cache file: {e}")
            raise
        for feature_name, data in np_cache.items():
            if data.dtype == object:
                for i in range(len(data)):
                    data_type = type(data[i])
                    if data_type in (str, bytes, np.ndarray):
                        data[i] = str(data[i])
                    else:
                        data[i] = str(data[i])
                try:
                    h5_cache.create_dataset(feature_name, data=data)
                except Exception as e:
                    logger.error(f"Error saving {feature_name} to cache: {e} with data {data}")
            else:
                h5_cache.create_dataset(feature_name, data=data)
        h5_cache.close()
                    
    def _transcode_pickled_images(self, ending_timestamp: Optional[int] = None):
        """
        Transcode pickled images into the desired format (e.g., raw or encoded images).
        """

        # Move the original file to a temporary location
        temp_path = self.path + ".temp"
        self._rename(self.path, temp_path)

        # Open the original container for reading
        original_container = av.open(temp_path, mode="r", format="matroska")
        original_streams = list(original_container.streams)

        # Create a new container
        new_container = av.open(self.path, mode="w", format="matroska")

        # Add existing streams to the new container
        d_original_stream_id_to_new_container_stream = {}
        for stream in original_streams:
            stream_feature = stream.metadata.get("FEATURE_NAME")
            if stream_feature is None:
                logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                continue
            # Determine encoding method based on feature type
            stream_encoding = self._get_encoding_of_feature(
                None, self.feature_name_to_feature_type[stream_feature]
            )
            stream_feature_type = self.feature_name_to_feature_type[stream_feature]
            stream_in_updated_container = self._add_stream_to_container(
                new_container, stream_feature, stream_encoding, stream_feature_type
            )

            # Preserve the stream metadata
            for key, value in stream.metadata.items():
                stream_in_updated_container.metadata[key] = value

            d_original_stream_id_to_new_container_stream[stream.index] = (
                stream_in_updated_container
            )

        # Initialize the number of packets per stream
        # Transcode pickled images and add them to the new container
        packets_muxed = 0
        for packet in original_container.demux(original_streams):

            def is_packet_valid(packet):
                return packet.pts is not None and packet.dts is not None

            if is_packet_valid(packet):
                original_stream = packet.stream
                new_stream = d_original_stream_id_to_new_container_stream[packet.stream.index]
                packet.stream = new_stream

                # Check if the ORIGINAL stream is using rawvideo, meaning it's a pickled stream
                if original_stream.codec_context.codec.name == "rawvideo":
                    logger.debug(f"Transcoding rawvideo packet from {original_stream.metadata.get('FEATURE_NAME')}")
                    data = pickle.loads(bytes(packet))

                    # Encode the image data with the new stream's encoding
                    new_packets = self._encode_frame(data, new_stream, packet.pts)

                    for new_packet in new_packets:
                        logger.debug(f"Muxing transcoded packet: {new_packet}")
                        new_container.mux(new_packet)
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
                flush_packets = stream.encode(None)
                logger.debug(f"Stream flush returned {len(flush_packets)} packets")
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

        # Reopen the new container for further writing new data
        # self.container_file = av.open(self.path, mode="a", format="matroska")

    def to_hdf5(self, path: Text):
        """
        convert the container file to hdf5 file
        """

        if not self.trajectory_data:
            self.load()

        # directly copy the cache file to the hdf5 file
        self._rename(self.cache_file_name, path)

    def _encode_frame(self, data: Any, stream: Any, timestamp: int) -> List[av.Packet]:
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
        logger.debug(f"Encoding {stream.metadata.get('FEATURE_NAME')} with {encoding}, feature_type: {feature_type}")
        
        if encoding == "ffv1" or encoding == "libaom-av1":
            logger.debug("Using video encoding path")
            if feature_type.dtype == "float32":
                frame = self._create_frame_depth(data, stream)
            else:
                frame = self._create_frame(data, stream)
            frame.pts = timestamp
            frame.dts = timestamp
            frame.time_base = stream.time_base
            logger.debug(f"Created frame: pts={frame.pts}, dts={frame.dts}")
            packets = stream.encode(frame)
            logger.debug(f"Stream encode returned {len(packets)} packets")
        else:
            logger.debug("Using rawvideo encoding path")
            packet = av.Packet(pickle.dumps(data))
            packet.dts = timestamp
            packet.pts = timestamp
            packet.time_base = stream.time_base
            packet.stream = stream
            logger.debug(f"Created raw packet: size={len(bytes(packet))}")

            packets = [packet]

        for packet in packets:
            packet.pts = timestamp
            packet.dts = timestamp
            packet.time_base = stream.time_base
        
        logger.debug(f"Returning {len(packets)} packets")
        return packets

    def _on_new_stream(self, new_feature, new_encoding, new_feature_type):
        if new_feature in self.feature_name_to_stream:
            return

        if not self.feature_name_to_stream:
            logger.debug(f"Creating a new stream for the first feature {new_feature}")
            self.feature_name_to_stream[new_feature] = self._add_stream_to_container(
                self.container_file, new_feature, new_encoding, new_feature_type
            )
        else:
            logger.debug(f"Adding a new stream for the feature {new_feature}")
            # Following is a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close(compact=False)

            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            self._rename(self.path, temp_path)

            # Open the original container for reading
            original_container = av.open(temp_path, mode="r", format="matroska")
            original_streams = list(original_container.streams)

            # Create a new container
            new_container = av.open(self.path, mode="w", format="matroska")

            # Add existing streams to the new container
            d_original_stream_id_to_new_container_stream = {}
            for stream in original_streams:
                stream_feature = stream.metadata.get("FEATURE_NAME")
                if stream_feature is None:
                    logger.debug(f"Skipping stream without FEATURE_NAME: {stream}")
                    continue
                stream_encoding = stream.codec_context.codec.name
                stream_feature_type = self.feature_name_to_feature_type[stream_feature]
                stream_in_updated_container = self._add_stream_to_container(
                    new_container, stream_feature, stream_encoding, stream_feature_type
                )
                # new_stream.options = stream.options
                for key, value in stream.metadata.items():
                    stream_in_updated_container.metadata[key] = value
                d_original_stream_id_to_new_container_stream[stream.index] = (
                    stream_in_updated_container
                )

            # Add new feature stream
            new_stream = self._add_stream_to_container(
                new_container, new_feature, new_encoding, new_feature_type
            )
            d_original_stream_id_to_new_container_stream[new_stream.index] = new_stream
            self.stream_id_to_info[new_stream.index] = StreamInfo(
                new_feature, new_feature_type, new_encoding
            )

            # Remux existing packets
            for packet in original_container.demux(original_streams):

                def is_packet_valid(packet):
                    return packet.pts is not None and packet.dts is not None

                if is_packet_valid(packet):
                    packet.stream = d_original_stream_id_to_new_container_stream[
                        packet.stream.index
                    ]
                    new_container.mux(packet)
                else:
                    pass

            original_container.close()
            self._remove(temp_path)

            # Reopen the new container for writing new data
            self.container_file = new_container
            self.feature_name_to_stream[new_feature] = new_stream
            self.is_closed = False

    def _add_stream_to_container(self, container, feature_name, encoding, feature_type):
        stream = container.add_stream(encoding)
        if encoding == "ffv1":
            stream.width = feature_type.shape[1]
            stream.height = feature_type.shape[0]
            
            # Set appropriate pixel format based on feature type for FFV1
            if len(feature_type.shape) == 3 and feature_type.shape[2] == 3:
                stream.pix_fmt = "yuv420p"  # FFV1 works better with YUV420p for RGB
            elif len(feature_type.shape) == 2 or (len(feature_type.shape) == 3 and feature_type.shape[2] == 1):
                stream.pix_fmt = "gray"
            else:
                stream.pix_fmt = "yuv420p"  # Default fallback
        
        if encoding == "libaom-av1":
            stream.width = feature_type.shape[1]
            stream.height = feature_type.shape[0]
            
            # Set appropriate pixel format based on feature type
            if len(feature_type.shape) == 3 and feature_type.shape[2] == 3:
                stream.pix_fmt = "yuv420p"  # AV1 typically uses YUV
            elif len(feature_type.shape) == 2 or (len(feature_type.shape) == 3 and feature_type.shape[2] == 1):
                stream.pix_fmt = "gray"
            else:
                stream.pix_fmt = "yuv420p"  # Default fallback
                
            stream.codec_context.options = {
                "g": "2",
                'crf': '30',  # Constant Rate Factor (quality)
            }

        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream

    def _create_frame(self, image_array, stream):
        image_array = np.array(image_array, dtype=np.uint8)
        encoding = stream.codec_context.codec.name
        
        # Determine the correct format based on array shape and codec
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB image
            if encoding == "libaom-av1" or encoding == "ffv1":
                # For AV1 and FFV1, convert RGB to YUV420p
                frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")
                frame = frame.reformat(format="yuv420p")
            else:
                frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            # Single channel image, squeeze the last dimension
            frame = av.VideoFrame.from_ndarray(image_array.squeeze(axis=2), format="gray")
        elif len(image_array.shape) == 2:
            # Grayscale image
            frame = av.VideoFrame.from_ndarray(image_array, format="gray")
        else:
            raise ValueError(f"Unsupported image array shape: {image_array.shape}")
        
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

    def _get_encoding_of_feature(
        self, feature_value: Any, feature_type: Optional[FeatureType]
    ) -> Text:
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
        data_shape = feature_type.shape
        if len(data_shape) >= 2 and data_shape[0] >= 100 and data_shape[1] >= 100:
            if self.lossy_compression:
                vid_coding = "libaom-av1"
            else:
                vid_coding = "ffv1"
        else:
            vid_coding = "rawvideo"
        return vid_coding

    def save_stream_info(self):
        # serialize and save the stream info
        with open(self.path + ".stream_info", "wb") as f:
            pickle.dump(self.stream_id_to_info, f)

    def load_stream_info(self):
        # load the stream info
        with open(self.path + ".stream_info", "rb") as f:
            self.stream_id_to_info = pickle.load(f)

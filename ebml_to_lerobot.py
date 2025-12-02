
from datetime import datetime
import numpy as np
import robodm
from robodm.loader.vla import (LoadingMode, RayVLALoader, SliceConfig,
                                create_slice_loader, create_trajectory_loader)
from robodm.utils.flatten import data_to_tf_schema

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

### TO-DO:
## High priority
# 1. Create a modality.json file using metadata info provided by robodm dataset. Consider having robodm also use the same modality.json?
# 1a. What is the fps of video_info data from robodm? Contain in robodm metadata
# 2. Look at how to deallocate loaded Ray data instance for reloading other robodm dataset instance 
#    for populating into the same LeRobot dataset instead of creating multiple folders
# 3. Set configs using yaml or argparse
## Low priority
# 1. To test with batch_size = 2 given a higher RAM CPU device
# 2. Only able to handle one text data in an entire LeRobot dataset now

### Load the trajectory into LeRobot format for training using GR00T
data_filename = "libero_10_no_noops_ep301-379"
batch_size = 1
task_labelling = ["instruction", "text", "task", "language"] # to identify "task" for LeRobot dataset
dataset_path = f"/home/ras/git_pkgs/robodm_yc_fork/data/libero_10_no_noops/{data_filename}.vla"
save_path = f"/home/ras/git_pkgs/robodm_yc_fork/data/lerobot_dataset/{data_filename}|" + datetime.now().strftime("%Y%m%d_%H:%M:%S")

### TO-DO: set the above configs by yaml or argparse

start_time = datetime.now()
print("Loading EBML dataset.")
VLALoader = create_trajectory_loader(path = dataset_path, batch_size=batch_size, return_type="numpy", shuffle=False)
from_metadata = VLALoader.schema()

from_metadata_keys = from_metadata.names
from_metadata_types = from_metadata.types

del from_metadata

data_list = VLALoader.get_batch() # this method is able to load dataset better and will automatically kill when out of memory
print("Successfully loaded EBML dataset.")
time_load_dataset = (datetime.now() - start_time).total_seconds()

# When batch_size = 1, data_list is a list of 1 dict of 12 keys each with 27647 trajectories for libero_10_no_noops_ep1-100.vla.
# len(data_list[0][column_names[0]]) = 27647
# num_trajectories = len(data_list[0][from_metadata_keys[0]])

"""
re: : 0.00 row [01:54, ? row/s]cks: 0; Resources: 0.0 CPU, 0.0B object store: : 0.00 row [01:54, ? row/s]
- limit=1: Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU, 10.1GB object store: - limit=1: Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU, 10.1GB object store: - Map(RayVLALoader._load_trajectory): Tasks: 4 [backpressured]; Queued blocks: 0- Map(RayVLALoader._load_trajectory): Tasks: 4 [backpressured]; Queued blocks: 0Running Dataset. Active & requested resources: 16/16 CPU, 91.2GB/4.1GB object stRunning Dataset. Active & requested resources: 16/16 CPU, 91.2GB/4.1GB object st✔️  Dataset execution finished in 115.87 seconds: : 0.00 row [01:55, ? row/s]                                                                                   
                                                                                ✔️  Dataset execution finished in 115.87 seconds: : 1.00 row [01:55, 116s/ row]
                                                                                - Map(RayVLALoader._load_trajectory): Tasks: 4 [backpressured]; Queued blocks: 0; Resources: 16.0 CPU, 81.0GB object store: : 1.00 row [01:55, 116s/ row]
- Map(RayVLALoader._load_trajectory): Tasks: 4 [backpressured]; Queued blocks: 0                                                                                - Map(RayVLALoader._load_trajectory): Tasks: 4 [backpressured]; Queued blocks: 0; Resources: 16.0 CPU, 81.0GB object store: : 1.00 row [01:55, 116s/ row]                                    

- limit=1: Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU, 10.1GB object store: - limit=1: Tasks: 0; Queued blocks: 0; Resources: 0.0 CPU, 10.1GB object store: : 1.00 row [01:55, 116s/ row]                                                   >>>                          
>>> (raylet) 
(raylet) [2025-09-08 18:42:49,188 E 10803 10803] (raylet) node_manager.cc:3069: 1 Workers (tasks / actors) killed due to memory pressure (OOM), 0 Workers crashed due to other reasons at node (ID: 1cd6e3ee8488c6321d3f29eb2fc507bdf4585e9f9c2c73dbd2a39d94, IP: 10.2.74.241) over the last time period. To see more information about the Workers killed on this node, use `ray logs raylet.out -ip 10.2.74.241`
(raylet) Refer to the documentation on how to address the out of memory issue: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html. Consider provisioning more memory on this node or reducing task parallelism by requesting more CPUs per task. To adjust the kill threshold, set the environment variable `RAY_memory_usage_threshold` when starting Ray. To disable worker killing, set the environment variable `RAY_memory_monitor_refresh_ms` to zero.

# libero_10_no_noops_ep1-100.vla requires about 20 GB memory to load into dictionary data structure. (27647 trajectories)
# 10,000 trajectories requires about 8 GB of memory. 
"""


### Sample LeRobot data: ['/home/ras/Isaac-GR00T/demo_data/robot_sim.PickNPlace']

'''
train_dataset is a <class 'gr00t.data.dataset.LeRobotSingleDataset'>. 
len(train_dataset) = 2096 trajectories. 
# type(dataset[0]['action']) = <class 'torch.Tensor'> # most are Tensor array, even if length = 1
type(dataset[-1]['task']) = str
train_dataset[0] is a dict, it gives the following output: 

{'state': array([[-1.14705776e-02,  1.21776660e-01,  4.22813668e-02,
        -8.63209367e-01, -1.44139529e-02, -3.01307627e-02,
        ...
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00]]), 
 'state_mask': array([[ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True, False, False,
        False, False, False, False, False, False, False, False, False,
        False]]), 
 'segmentation_target': array([0., 0.]), 
 'segmentation_target_mask': array([0.]), 
 'has_real_action': array(True), 
 'action': array([[-0.99999999,  0.75398925,  1.00000001, -0.65923027, -0.86737372,
        ...
        0.        ,  0.        ]]), 
 'action_mask': array([[ True, 
        ...
        [ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True, False,
        False, False, False, False, False]]), 
 'eagle_content': {'image_inputs': [<PIL.Image.Image image mode=RGB size=224x224 at 0x723F8677D180>], 
        'video_inputs': None, 'text_list': ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image-1>
        pick the pear from the counter and place it in the plate<|im_end|>\n<|im_start|>assistant\n']}, 
 'embodiment_id': 31}     
'''

# input target metadata
# to_metadata = ['action',
#             'discount',
#             'is_first',
#             'is_last',
#             'is_terminal',
#             'language_instruction',
#             'observation/image',
#             'observation/joint_state',
#             'observation/state',
#             'observation/wrist_image',
#             'reward'
#             ]

# def create_lerobot_dataset(
#     raw_dir: Path,
#     repo_id: str = None,
#     local_dir: Path = None,
#     push_to_hub: bool = False,
#     fps: int = None,
#     robot_type: str = None,
#     use_videos: bool = True,
#     image_writer_process: int = 5,
#     image_writer_threads: int = 10,
#     keep_images: bool = True,
# ):

### Load tfds data - OpenX
# builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
# features = generate_features_from_raw(builder, use_videos)
# filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
# raw_dataset = (
#     builder.as_dataset(split="train")
#     .filter(filter_fn)
#     .map(partial(transform_raw_dataset, dataset_name=dataset_name))
# )

# class type Dtype to match <class 'ray.air.util.tensor_extensions.pandas.TensorDtype'>
class Dtype():
    def __init__(self, dtype, shape):
        self._dtype = dtype
        self._shape = shape

# populate features using metadata loaded from robodm for LeRobot
num_text_data = 0
features = {}
dtype_feature_dict = {'bool': [], 'float32': [], 'float64': [], 'int8': [], 'int32': [], 'video': []} # standard LeRobot float32, int64?
dtype_num_features = 0
data_features_popped_list = []
for i, feature in enumerate(from_metadata_keys):

    # episode_index is not a feature in the dataset, but is instead metadata
    # depth data is omitted from dataset
    if 'episode_index' in feature or 'depth' in feature:
        data_features_popped_list.append(feature)
        continue

    # LeRobot dataset handles string data type by assigning to task
    if any(word in feature for word in task_labelling):
        task_label = feature
        num_text_data += 1
        data_features_popped_list.append(feature)
        continue

    dtype = from_metadata_types[i].to_pandas_dtype()
    # Remove the length of the trajectory / episode from shape
    if len(dtype._shape) == 1:
        dtype._shape = (1,)
    elif len(dtype._shape) > 1:
        dtype._shape = dtype._shape[1:]
    else:
        raise ValueError("Invalid shape detected, the length of a dataset key is less than 1. Please check the dataset input!")
    names = None
    
    # filters out the numpy class from dtype
    features[feature] = {'dtype': str(dtype._dtype).split("'")[1].split('.')[-1].replace('_',''), 'shape': dtype._shape, 'names': names}

    # conversion to pyarrow array for creating LeRobot dataset. Note: NumPy is more memory efficient for computation, whereas PyArrow is more memory efficient for data transport and storage
    # Not in use because i.e. float32 vs. float16 will not be differentiated
    # features[feature] = {'dtype': str(pa.from_numpy_dtype(dtype._dtype)), 'shape': dtype._shape, 'names': names}

    if 'cam' in feature or 'image' in feature:
        dtype = Dtype('video', data_list[0][feature][0].shape) # list index out of range because of language instruction?
        names = ['height', 'width', 'channel']
        features[feature] = {'dtype': dtype._dtype, 'shape': dtype._shape, 'names': names}
        # To compress into mp4 format via LeRobot dataset using ffmpeg to convert frames stored as png into mp4 videos.
        features[feature]['video_info'] = {'video.fps': 10.0, # TO-DO: to modify according to metadata accordingly
                                           'video.codec': 'h264',
                                           'video.pix_fmt': 'yuv420p',
                                           'video.is_depth_map': False,
                                           'video.has_audio': False}
    if features[feature]['dtype'] in dtype_feature_dict:
        dtype_feature_dict[features[feature]['dtype']].append(feature)
        dtype_num_features += 1
print("Processed metadata for LeRobot dataset.")

# check that number of features captured is consistent
if not len(features) == dtype_num_features:
    raise ValueError("Number of features captured for conversion not consistent. Please check or expand on data type accepted in variable: dtype_feature_dict.")

# dataset.features['observation.images.cam_high'] 
# {'dtype': 'video', 
#  'shape': (480, 640, 3), 
#  'names': ['height', 'width', 'channel'], 
#  'video_info': {'video.fps': 50.0, 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}
# dataset.features['observation.state'] 
# {'dtype': 'float32', 
#  'shape': (14,), 
#  'names': {'motors': ['left_waist', 'left_shoulder']}}

### configs to handle argparse in beginning
robot_type = "mobile_manipulator"
fps = 10
if fps is None:
    fps = 10
use_videos = True
image_writer_process = 5
image_writer_threads = 10
keep_images = True
push_to_hub = False

print("Saving dataset to path: ", save_path)
start_time = datetime.now()

# Declare LeRobot dataset to save to 
lerobot_dataset = LeRobotDataset.create(
    repo_id=data_filename,
    robot_type=robot_type,
    root=save_path,
    fps=int(fps),
    use_videos=use_videos,
    features=features,
    image_writer_threads=image_writer_threads,
    image_writer_processes=image_writer_process,
)


# below faces the error: 
'''
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py", line 1043, in create
    obj.hf_dataset = obj.create_hf_dataset()
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py", line 611, in create_hf_dataset
    features = get_hf_features_from_features(self.features)
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/utils.py", line 373, in get_hf_features_from_features
    length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
  File "<string>", line 5, in __init__
  File "/home/ras/py3.10/lib/python3.10/site-packages/datasets/features/features.py", line 528, in __post_init__
    self.pa_type = string_to_arrow(self.dtype)
  File "/home/ras/py3.10/lib/python3.10/site-packages/datasets/features/features.py", line 261, in string_to_arrow
    raise ValueError(
ValueError: Neither <class 'numpy.float32'> nor <class 'numpy.float32'>_ seems to be a pyarrow data type. Please make sure to use a correct data type, see: https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions

Other issue 23 Sept 2025 2301: 
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py", line 780, in add_frame
    validate_frame(frame, self.features)
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/utils.py", line 754, in validate_frame
    raise ValueError(error_message)
ValueError: Feature mismatch in `frame` dictionary:
Extra features: {'episode_index'}
The feature 'is_terminal' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'numpy.bool_'>' provided instead.
The feature 'is_first' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'numpy.bool_'>' provided instead.
The feature 'is_last' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'numpy.bool_'>' provided instead.
The feature 'observation.joint_state' of dtype 'float32' is not of the expected dtype 'float'.
The feature 'observation.state' of dtype 'float32' is not of the expected dtype 'float'.
The feature 'discount' is not a 'np.ndarray'. Expected type is 'float', but type '<class 'numpy.float32'>' provided instead.
The feature 'action' of dtype 'float32' is not of the expected dtype 'float'.
The feature 'reward' is not a 'np.ndarray'. Expected type is 'float', but type '<class 'numpy.float32'>' provided instead.

24 Sept 2025 1150:
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py", line 780, in add_frame
    validate_frame(frame, self.features)
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/utils.py", line 754, in validate_frame
    raise ValueError(error_message)
ValueError: The feature 'is_terminal' is not a 'np.ndarray'. Expected type is 'bool_', but type '<class 'numpy.bool_'>' provided instead.
The feature 'is_first' is not a 'np.ndarray'. Expected type is 'bool_', but type '<class 'numpy.bool_'>' provided instead.
The feature 'is_last' is not a 'np.ndarray'. Expected type is 'bool_', but type '<class 'numpy.bool_'>' provided instead.
The feature 'discount' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'numpy.float32'>' provided instead.
The feature 'reward' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'numpy.float32'>' provided instead.

24 Sept 2025 1332:
Traceback (most recent call last):
  File "<stdin>", line 26, in <module>
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py", line 780, in add_frame
    validate_frame(frame, self.features)
  File "/home/ras/py3.10/lib/python3.10/site-packages/lerobot/datasets/utils.py", line 754, in validate_frame
    raise ValueError(error_message)
ValueError: The feature 'reward' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'float'>' provided instead.
The feature 'is_first' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'bool'>' provided instead.
The feature 'is_last' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'bool'>' provided instead.
The feature 'discount' is not a 'np.ndarray'. Expected type is 'float32', but type '<class 'float'>' provided instead.
The feature 'is_terminal' is not a 'np.ndarray'. Expected type is 'bool', but type '<class 'bool'>' provided instead.

'''

### to test:
# data = VLALoader.iter_batches(batch_size = 1)
# data = VLALoader.iter_rows()

for data_batch in data_list:
    # for key, value in data_batch.items():
    #     data_batch[key] = pa.array(value) # only works for 1D array, not able to process image data
    if len(data_batch) - len(data_features_popped_list) != len(features): # remove episode and remove task / language_instruction
        raise ValueError("The extracted metadata number of features do not match the dataset number of features.")
    for index in range(len(data_batch[from_metadata_keys[0]])):
        
        # Check that if episode has changed, save dataset
        if data_batch['episode_index'][index] > data_batch['episode_index'][index - 1]: 
            lerobot_dataset.save_episode() # final iteration save
            print(f"Saved episode {data_batch['episode_index'][index - 1]}.")
        
        # populate frame with data
        data_dict = {}
        for d_key, feature_list in dtype_feature_dict.items(): 
            for feature in feature_list:
                if features[feature]['shape'] == (1,):
                    data_dict[feature] = np.array([data_batch[feature][index]])
                    # if 'float' in d_key:
                    #     data_dict[feature] = float(data_batch[feature][index])
                    # elif 'bool' in d_key:
                    #     data_dict[feature] = bool(data_batch[feature][index])
                    # elif 'int' in d_key:
                    #     data_dict[feature] = int(data_batch[feature][index])
                else:
                    data_dict[feature] = data_batch[feature][index]
        # print("Adding frame...")
        lerobot_dataset.add_frame(data_dict, task=data_batch[task_label][index].item().decode('utf-8'))
        
        ### old method does not account for variation in data type
        # lerobot_dataset.add_frame(
        #     {
        #         f"{key}": np.array(value[index]) # if len(value[index].shape) != 0 else np.array([value[index]])
        #         for key, value in data_batch.items()
        #         if not any(word in key for word in task_labelling)
        #         # if "depth" not in key and "instruction" not in key and "text" not in key and "task" not in key
        #     },
        #     task=data_batch[task_label][index]
        # )
    lerobot_dataset.save_episode() # final iteration save
    print(f"Saved final episode {data_batch['episode_index'][-1]}.")

time_save_dataset = (datetime.now() - start_time).total_seconds()
# Quantify time taken to process entire dataset. 
print("Time to load EBML dataset instance:", time_load_dataset, "seconds.")
print("Time to save LeRobot dataset instance:", time_save_dataset, "seconds.")

        # image_dict = {
        #     f"observation.{key}": value[index]
        #     for key, value in data_batch.items()
        #     if "depth" not in key and any(x in key for x in ["image", "rgb", "cam", "img"])
        # }

        # for feature in from_metadata_keys:

        # lerobot_dataset.add_frame(
        #     {
        #         **image_dict,
        #         "observation.state": data_batch["observation.state"][index],
        #         "action": data_batch["action"][index],
        #         "discount": data_batch["discount"][index],
        #         "is_first": data_batch["is_first"][index],
        #         "is_last": data_batch["is_last"][index],
        #         "is_terminal": data_batch["is_terminal"][index],
        #         "observation": data_batch["observation"][index],
        #         "reward": data_batch["reward"][index],
        #     },
        #     task=data_batch["language_instruction"][index],
        # )

### adapt from: https://github.com/Tavish9/any4lerobot/blob/main/openx2lerobot/openx_rlds.py
# def save_as_lerobot_dataset(lerobot_dataset: LeRobotDataset, raw_dataset: tf.data.Dataset, **kwargs):
#     for episode in raw_dataset.as_numpy_iterator():
#         traj = episode["steps"]
#         for i in range(traj["action"].shape[0]):
#             image_dict = {
#                 f"observation.images.{key}": value[i]
#                 for key, value in traj["observation"].items()
#                 if "depth" not in key and any(x in key for x in ["image", "rgb"])
#             }
#             lerobot_dataset.add_frame(
#                 {
#                     **image_dict,
#                     "observation.state": traj["proprio"][i],
#                     "action": traj["action"][i],
#                 },
#                 task=traj["task"][0].decode(),
#             )
#         lerobot_dataset.save_episode()

if push_to_hub:
    assert repo_id is not None
    tags = ["LeRobot", data_filename, "robodm"]
    # if dataset_name in OXE_DATASET_CONFIGS:
    #     tags.append("openx")
    if robot_type != "unknown":
        tags.append(robot_type)
    lerobot_dataset.push_to_hub(
        tags=tags,
        private=False,
        push_videos=True,
        license="apache-2.0",
    )
    print("LeRobot dataset pushed to hub.")


print("Closing application.")
exit()

### Loading LeRobot dataset
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

### To load local dataset


### To download online dataset and load
repo_id = "lerobot/aloha_mobile_cabinet"
dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23]) # <class 'lerobot.datasets.lerobot_dataset.LeRobotDataset'>
print(dataset.meta) # <class 'lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata'>
'''
LeRobotDatasetMetadata({
    Repository ID: 'lerobot/aloha_mobile_cabinet',
    Total episodes: '85',
    Total frames: '127500',
    Features: '['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'observation.state', 'observation.effort', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.done', 'index', 'task_index']',
})',
'''
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# Then we grab all the image frames from the first camera:
camera_key = dataset.meta.camera_keys # returns list of all camera data
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]
print(type(frames[0]))
print(frames[0].shape)

print(dataset.features[camera_key])
print(dataset.features[camera_key]["shape"])

# dataset.features['observation.images.cam_high'] # camera feature dict is different
# {'dtype': 'video', 
#  'shape': (480, 640, 3), 
#  'names': ['height', 'width', 'channel'], 
#  'video_info': {'video.fps': 50.0, 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}
# dataset.features['observation.state'] # default feature dict consisting dtype, shape, names
# {'dtype': 'float32', 
#  'shape': (14,), 
#  'names': {'motors': ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']}}
# dataset.features['frame_index']
# {'dtype': 'int64', 'shape': (1,), 'names': None}
# dataset.features['task_index']
# {'dtype': 'int64', 'shape': (1,), 'names': None}
# dataset.features['index']
# {'dtype': 'int64', 'shape': (1,), 'names': None}
# dataset.features['next.done']
# {'dtype': 'bool', 'shape': (1,), 'names': None}

# type(dataset.features) = <class 'dict'>
# len(dataset.features) = 12

# ds_meta = LeRobotDatasetMetadata(repo_id)
# print(ds_meta.features)

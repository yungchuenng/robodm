
import numpy as np
import robodm
from robodm.loader import RLDSLoader

# TO-DO: Allow argparse for all possible inputs, load format and save format
# TO-DO: OOP
# TO-DO: one script to convert all types of dataset into VLA
# TO-DO: implement metadata mapping - from_metadata to to_metadata dictionary key-value, including allow input of data dtype with enumerate mapping to i.e. np.float32
# TO-DO: implement method to load data from mp4 / other video formats and label frame_index accordingly
# TO-DO: if given timestamp, directly paste timestamp in, else if given frequency, manually compute time starting from t=0, else warn if there is no timestamp within from_metadata
# TO-DO: Check based on RAM available and datasize of one trajectory, how many trajectories can be loaded
# TO-DO: Allow to specify the data for loading to use in training, batch_size and what files to load
# TO-DO: Convert EBML back into LeRobot format
# TO-DO: How to free up unused RAM?

# dataset_path = "/home/ras/git_pkgs/LIBERO/libero/datasets/openvla/modified_libero_rlds/libero_10_no_noops/1.0.0"
dataset_path = "/home/ras/git_pkgs/LIBERO/libero/datasets/openvla/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
rlds_loader = RLDSLoader(path=dataset_path, split="all", batch_size=1, shuffle_buffer=1,shuffling=False)

### loads the entire dataset at once - 43 GB memory
# rlds_loader.batch_size = rlds_loader.__len__()
# ds = rlds_loader.get_batch()

# either input metadata or extract metadata from features.json of RLDS
from_metadata = ['action',
                'discount',
                'is_first',
                'is_last',
                'is_terminal',
                'language_instruction',
                'observation',
                'reward'
                ]

# remove 68 of 500 LIBERO-Spatial demonstrations, 
# remove 46 of 500 LIBERO-Object demonstrations, 
# remove 72 of 500 LIBERO-Goal demonstrations, 
# remove 121 of 500 LIBERO-Long (10) demonstrations ===> Thus, 379 episodes
### Sample RLDS data: libero_10_no_noops (for the case where batch_size=1) (only split available is train)
# Takes up 43 GB of memory to load the entire dataset in dictionary. 
# type(ds) = list of 379 tasks
# type(ds[0]) = list of 214 trajectories
# type(ds[0][0]) = dictionary of format: 
# image is of shape: (256, 256, 3)
'''
{'action': array([ 0.01607143, 0., -0.,  0.,  0., -0., -1.], dtype=float32), 
 'discount': array(1., dtype=float32), 
 'is_first': array(True), 
 'is_last': array(False), 
 'is_terminal': array(False), 
 'language_instruction': array(b'put the white mug on the left plate and put the yellow and white mug on the right plate', dtype=object), 
 'observation': {'image': array([[[ 13,  13,  13],
                        [ 13,  13,  13],
                        [ 13,  13,  13],
                        ...,
                        [116, 117, 112],
                        [115, 116, 111],
                        [114, 115, 110]],

                       [[ 13,  13,  13],
                        [ 13,  13,  13],
                        [ 13,  13,  13],
                        ...,
                        [116, 117, 112],
                        [115, 116, 111],
                        [114, 115, 110]],

                       [[ 13,  13,  13],
                        [ 13,  13,  13],
                        [ 13,  13,  13],
                        ...,
                        [116, 117, 112],
                        [115, 116, 111],
                        [114, 115, 110]],

                       ...,

                       [[ 83,  64,  50],
                        [ 82,  63,  49],
                        [ 81,  62,  48],
                        ...,
                        [ 76,  54,  41],
                        [ 75,  53,  40],
                        [ 74,  52,  39]],

                       [[ 78,  58,  47],
                        [ 76,  56,  45],
                        [ 75,  56,  42],
                        ...,
                        [ 79,  57,  43],
                        [ 79,  57,  43],
                        [ 79,  57,  43]],

                       [[ 75,  55,  44],
                        [ 73,  53,  42],
                        [ 73,  54,  40],
                        ...,
                        [ 79,  57,  43],
                        [ 79,  57,  43],
                        [ 81,  59,  45]]], dtype=uint8), 
                 'joint_state': array([ 0.00380528, -0.14113528,  0.01111517, -2.4312031 ,  0.00360135, 2.2328262 ,  0.7972417 ], dtype=float32), 
                 'state': array([-5.3380046e-02,  7.0296312e-03,  6.7832810e-01,  3.1407692e+00, 1.7593271e-03, -8.9944184e-02,  3.8788661e-02, -3.8787212e-02], dtype=float32), 
                 'wrist_image': array([[[90, 65, 45],
                        [93, 68, 48],
                        [93, 68, 48],
                        ...,
                        [91, 68, 54],
                        [91, 68, 54],
                        [90, 67, 53]],

                       [[91, 66, 46],
                        [93, 68, 48],
                        [95, 70, 50],
                        ...,
                        [90, 67, 53],
                        [89, 66, 52],
                        [89, 66, 52]],

                       [[94, 68, 51],
                        [96, 70, 53],
                        [97, 71, 54],
                        ...,
                        [93, 67, 52],
                        [93, 67, 52],
                        [93, 67, 52]],

                       ...,

                       [[54, 54, 54],
                        [54, 54, 54],
                        [54, 54, 54],
                        ...,
                        [55, 55, 55],
                        [55, 55, 55],
                        [55, 55, 55]],

                       [[54, 54, 54],
                        [54, 54, 54],
                        [54, 54, 54],
                        ...,
                        [55, 55, 55],
                        [55, 55, 55],
                        [55, 55, 55]],

                       [[54, 54, 54],
                        [54, 54, 54],
                        [54, 54, 54],
                        ...,
                        [55, 55, 55],
                        [55, 55, 55],
                        [55, 55, 55]]], dtype=uint8)}, 
 'reward': array(0., dtype=float32)}
'''
# type(ds[0][-1]) = dictionary of format: 
# same as above, except the following key-items:
'''
'is_first': array(False), 
'is_last': array(True), 
'is_terminal': array(True)
'''
# (for the case where batch_size=10)
# len(ds_batch) = 10 (list of 10 lists)
# len(ds_batch[0]) = 214 (list of 214 trajectories)
###

# Set filename for first dataset, saving at every 100 tasks
max_ep_length = rlds_loader.__len__()
if max_ep_length > 100: ep_end = "100"
else: ep_end = max_ep_length

# Create robodm EBML dataset to save to
ebml_ds = robodm.Trajectory(path=f"/home/ras/git_pkgs/robodm_yc_fork/data/libero_spatial_no_noops_ep1-{ep_end}.vla", 
                                mode="w", 
                                video_codec="libx265",  # auto
                                codec_options={
                                    "crf": "23",        # Quality setting (lower = higher quality)
                                    "preset": "fast"    # Encoding speed
                                    }
                                )
print("Initialising EBML dataset to save video using codec: libx265")

# input target metadata
to_metadata = ['action',
            'discount',
            'is_first',
            'is_last',
            'is_terminal',
            'language_instruction',
            'observation/image',
            'observation/joint_state',
            'observation/state',
            'observation/wrist_image',
            'reward'
            ]

# index 0
ds = rlds_loader.get_batch()
episode = ds[0]
ep_idx = 0 # Episode index will begin at 1

# loading episodes one at a time
for index in range(rlds_loader.__len__()):
    ep_idx += 1

    # save file if task interval is reached
    if not(index % 100):
        ebml_ds.close()
        print("EBML dataset saved.")
        
        # rename file for subsequent dataset according to task interval
        ep_end = index + 100
        if ep_end > max_ep_length: ep_end = max_ep_length
        ebml_ds = robodm.Trajectory(path=f"/home/ras/git_pkgs/robodm_yc_fork/data/libero_spatial_no_noops_ep{ep_idx}-{ep_end}.vla", 
                                mode="w", 
                                video_codec="libx265",  # auto
                                codec_options={
                                    "crf": "23",        # Quality setting (lower = higher quality)
                                    "preset": "fast"    # Encoding speed
                                    }
                                )

    print("Processing episode:", ep_idx)
    for trajectory in episode:

        # Add data to the robodm EBML dataset
        ebml_ds.add("episode_index", np.array(ep_idx, dtype=np.int32))
        ebml_ds.add(to_metadata[0], np.array(trajectory[from_metadata[0]]))
        ebml_ds.add(to_metadata[1], np.array(trajectory[from_metadata[1]]))
        ebml_ds.add(to_metadata[2], np.array(trajectory[from_metadata[2]]))
        ebml_ds.add(to_metadata[3], np.array(trajectory[from_metadata[3]]))
        ebml_ds.add(to_metadata[4], np.array(trajectory[from_metadata[4]]))
        ebml_ds.add(to_metadata[5], np.array(trajectory[from_metadata[5]]))
        ebml_ds.add(to_metadata[6], np.array(trajectory[from_metadata[6]]['image']))
        ebml_ds.add(to_metadata[7], np.array(trajectory[from_metadata[6]]['joint_state']))
        ebml_ds.add(to_metadata[8], np.array(trajectory[from_metadata[6]]['state']))
        ebml_ds.add(to_metadata[9], np.array(trajectory[from_metadata[6]]['wrist_image']))
        ebml_ds.add(to_metadata[10], np.array(trajectory[from_metadata[7]]))

    try:
        episode = rlds_loader.__next__()[0]
    except:
        print('Index', rlds_loader.index, 'has reached the end of the dataset.')
        # raise StopIteration # already raised in above code
        break

# save EBML dataset when end of dataset is reached
ebml_ds.close()
print("EBML dataset saved.")

exit()

### Load the trajectory for training
print("Loading EBML dataset.")
dataset_path = "/home/ras/git_pkgs/robodm_yc_fork/data/libero_10_no_noops_ep1-100.vla"
ebml_ds = robodm.Trajectory(path=dataset_path, mode="r")
data = ebml_ds.load() # data_slice=slice(0,100)
data1 = ebml_ds.load()

'''
from robodm.loader.vla import (LoadingMode, RayVLALoader, SliceConfig, \
                                create_slice_loader, create_trajectory_loader)
from robodm.utils.flatten import data_to_tf_schema

VLALoader = create_trajectory_loader(path = path_to_vla_dataset, batch_size=1, return_type="numpy", shuffle=False)
data_list = VLALoader.get_batch() # this method is able to load dataset better and will automatically kill when out of memory

When batch_size = 1, data_list is a list of 1 dict of 12 keys each with 27647 trajectories for libero_10_no_noops_ep1-100.vla.

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
"""

data = VLALoader._load_trajectory(path_to_vla_dataset) # not enough memory. Returns dict
data = VLALoader._extract_slices(path_to_vla_dataset) # Will not have enough memories. Returns list of dicts and in each dict, contains slice_length no. of trajectories

def create_trajectory_loader(
    path: Text,
    batch_size: int = 1,
    return_type: str = "numpy",
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    **kwargs,
) -> RayVLALoader:
    """Create a loader for complete trajectories."""


def create_slice_loader(
    path: Text,
    slice_length: int = 100,
    batch_size: int = 1,
    return_type: str = "numpy",
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    min_slice_length: Optional[int] = None,
    stride: int = 1,
    random_start: bool = True,
    overlap_ratio: float = 0.0,
    """Creates a loader for trajectory slices"""

    def _load_trajectory(self, item) -> Dict[str, Any]:
    def _extract_slices(self, item) -> List[Dict[str, Any]]:

'''

for metadata in to_metadata:
    data[metadata] = numpy.append(data[metadata], data1[metadata])

# libero_10_no_noops_ep1-100.vla requires about 20 GB memory to load into dictionary data structure. (27647 trajectories)
# 10,000 trajectories requires about 8 GB of memory. 
# print("Able to load trajectory data:")

# print(f"observation state shape: {data['observation/state'].shape}")
# print(f"observation ee state shape: {data['observation/states/ee_state'].shape}")
# print(f"observation joint state shape: {data['observation/states/joint_state'].shape}")
# print(f"observation gripper state shape: {data['observation/states/gripper_state'].shape}")
# print(f"action shape: {data['action'].shape}")
# print(f"timestamp shape: {data['timestamp'].shape}")
# print(f"observation state: {data['observation/state'][0]}")
# print(f"observation ee state: {data['observation/states/ee_state'][0]}")
# print(f"observation joint state: {data['observation/states/joint_state'][0]}")
# print(f"observation gripper state: {data['observation/states/gripper_state'][0]}")
# print(f"action: {data['action'][0]}")
# print(f"front camera shape: {data['sensors/camera/front/rgb'].shape}")
# print(f"wrist camera shape: {data['sensors/camera/wrist/rgb'].shape}")
# print(f"front camera frame 0 dtype: {data['sensors/camera/front/rgb'][0].dtype}")
# print(f"wrist camera frame 0 dtype: {data['sensors/camera/wrist/rgb'][0].dtype}")
# print(f"front camera frame 0: {data['sensors/camera/front/rgb'][0]}")


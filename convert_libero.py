import cv2
import numpy as np
import robodm
from datasets import load_dataset

def get_chunk_folder(episode_index):
    if 0 <= episode_index <= 999:
        return "chunk-000"
    elif 1000 <= episode_index <= 1999:
        return "chunk-001"
    elif 2000 <= episode_index <= 2999:
        return "chunk-002"
    elif 3000 <= episode_index <= 3920:
        return "chunk-003"
    else:
        raise ValueError(f"Episode index {episode_index} out of expected range")

# Variables to hold current open video captures to avoid reopening
current_episode = None
cap_image = None
cap_wrist = None

def open_videos_for_episode(episode_index):
    chunk_folder = get_chunk_folder(episode_index)
    episode_str = f"episode_{episode_index:06d}.mp4"

    #download video first(https://huggingface.co/datasets/IPEC-COMMUNITY/libero_90_no_noops_lerobot)
    video_path_image = f"libero_90_no_noops_lerobot/videos/{chunk_folder}/observation.images.image/{episode_str}"
    video_path_wrist = f"libero_90_no_noops_lerobot/videos/{chunk_folder}/observation.images.wrist_image/{episode_str}"

    cap_img = cv2.VideoCapture(video_path_image)
    cap_wri = cv2.VideoCapture(video_path_wrist)

    if not cap_img.isOpened():
        raise RuntimeError(f"Failed to open image video: {video_path_image}")
    if not cap_wri.isOpened():
        raise RuntimeError(f"Failed to open wrist video: {video_path_wrist}")

    return cap_img, cap_wri

# Only process episode 0
cap_image, cap_wrist = open_videos_for_episode(0)

def read_frame(cap, frame_idx):
    # Seek to desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


ds = load_dataset("IPEC-COMMUNITY/libero_90_no_noops_lerobot", split="train")
trajectory = robodm.Trajectory(path="data/libero_90_no_noops_lerobot.vla", mode="w", video_codec="libx265")

for entry in ds:
    ep_idx = entry["episode_index"]
    ###
    #for testing
    if ep_idx != 0:
        break 
    ###
    frm_idx = entry["frame_index"]

    # If new episode, open corresponding videos and close previous if any
    '''
    if ep_idx != current_episode:
        if cap_image is not None:
            cap_image.release()
        if cap_wrist is not None:
            cap_wrist.release()
        cap_image, cap_wrist = open_videos_for_episode(ep_idx)
        current_episode = ep_idx
    '''
    # Read the frames for this timestep
    front_camera = read_frame(cap_image, frm_idx)
    wrist_camera = read_frame(cap_wrist, frm_idx)

    # Add video frames
    trajectory.add("sensors/camera/front/rgb", front_camera)
    trajectory.add("sensors/camera/wrist/rgb", wrist_camera)

    # Add metadata to the trajectory
    trajectory.add("observation/state", np.array(entry["observation.state"], dtype=np.float32))
    trajectory.add("observation/states/ee_state", np.array(entry["observation.states.ee_state"], dtype=np.float32))
    trajectory.add("observation/states/joint_state", np.array(entry["observation.states.joint_state"], dtype=np.float32))
    trajectory.add("observation/states/gripper_state", np.array(entry["observation.states.gripper_state"], dtype=np.float32))
    trajectory.add("action", np.array(entry["action"], dtype=np.float32))
    trajectory.add("timestamp", np.array(entry["timestamp"], dtype=np.float32))
    trajectory.add("frame_index", np.array(frm_idx, dtype=np.int32))
    trajectory.add("episode_index", np.array(ep_idx, dtype=np.int32))
    trajectory.add("index", np.array(entry["index"], dtype=np.int32))
    trajectory.add("task_index", np.array(entry["task_index"], dtype=np.int32))

# Clean up video captures
if cap_image is not None:
    cap_image.release()
if cap_wrist is not None:
    cap_wrist.release()

trajectory.close()
print("Trajectory saved with synchronized video and metadata.")

# Load the trajectory for training
trajectory = robodm.Trajectory(path="/tmp/libero_full.vla", mode="r")
data = trajectory.load()

print("Loaded trajectory data:")
print(f"observation state shape: {data['observation/state'].shape}")
print(f"observation ee state shape: {data['observation/states/ee_state'].shape}")
print(f"observation joint state shape: {data['observation/states/joint_state'].shape}")
print(f"observation gripper state shape: {data['observation/states/gripper_state'].shape}")
print(f"action shape: {data['action'].shape}")
print(f"timestamp shape: {data['timestamp'].shape}")
print(f"observation state: {data['observation/state'][0]}")
print(f"observation ee state: {data['observation/states/ee_state'][0]}")
print(f"observation joint state: {data['observation/states/joint_state'][0]}")
print(f"observation gripper state: {data['observation/states/gripper_state'][0]}")
print(f"action: {data['action'][0]}")
print(f"front camera shape: {data['sensors/camera/front/rgb'].shape}")
print(f"wrist camera shape: {data['sensors/camera/wrist/rgb'].shape}")
print(f"front camera frame 0 dtype: {data['sensors/camera/front/rgb'][0].dtype}")
print(f"wrist camera frame 0 dtype: {data['sensors/camera/wrist/rgb'][0].dtype}")
print(f"front camera frame 0: {data['sensors/camera/front/rgb'][0]}")

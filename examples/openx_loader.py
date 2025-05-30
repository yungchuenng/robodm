import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from fog_x.loader import RLDSLoader
import fog_x
import threading
import time

def process_data(data_traj, dataset_name, index, destination_dir, video_codec):
    try:
        data_traj = data_traj[0]
        fog_x.Trajectory.from_list_of_dicts(
            data_traj, path=f"{destination_dir}/{dataset_name}/output_{index}.vla",
            video_codec=video_codec
        )
        print(f"Processed data {index}")
        return index, True
    except Exception as e:
        print(f"Failed to process data {index}: {e}")
        return index, False

def main():
    parser = argparse.ArgumentParser(description="Convert OpenX datasets to VLA format")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to convert")
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset")
    parser.add_argument("--destination_dir", type=str, help="Destination directory for VLA files")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to process")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--video_codec", type=str, default="auto", 
                       choices=["auto", "rawvideo", "h264", "h265", "libaom-av1", "ffv1"],
                       help="Video codec to use for encoding")
    
    args = parser.parse_args()
    
    # Create destination directory
    os.makedirs(f"{args.destination_dir}/{args.dataset_name}", exist_ok=True)
    
    # Load dataset
    loader = RLDSLoader(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        shuffle=False
    )
    
    trajectories = loader.load_trajectories()
    if args.max_episodes:
        trajectories = trajectories.take(args.max_episodes)
    
    # Process trajectories in parallel
    futures = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for i, data_traj in enumerate(trajectories):
            future = executor.submit(
                process_data, data_traj, args.dataset_name, i, 
                args.destination_dir, args.video_codec
            )
            futures.append(future)
        
        # Collect results
        success_count = 0
        total_count = 0
        for future in as_completed(futures):
            index, success = future.result()
            total_count += 1
            if success:
                success_count += 1
    
    print(f"Conversion complete: {success_count}/{total_count} trajectories processed successfully")

if __name__ == "__main__":
    main()
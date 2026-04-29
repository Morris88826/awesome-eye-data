import os
import cv2
import tqdm
import argparse
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

def load_annotations(annotation_paths):
    validity_path = annotation_paths['validity']
    validity_df = pd.read_csv(validity_path, sep=";", usecols=["FRAME", "VALIDITY"])

    gaze_path = annotation_paths['gaze_vector']
    gaze_df = pd.read_csv(gaze_path, sep=";", usecols=["FRAME", "X", "Y", "Z"])

    pupilCenter_path = annotation_paths['pupil_center']
    pupilCenter_df = pd.read_csv(pupilCenter_path, sep=";", usecols=["FRAME", "CENTER X", "CENTER Y"])

    df_all = (
        validity_df
        .merge(gaze_df, on="FRAME", how="left")
        .merge(pupilCenter_df, on="FRAME", how="left")
    )

    # rename columns
    df_all.rename(columns={
        "VALIDITY": "validity",
        "X": "gaze_x",
        "Y": "gaze_y",
        "Z": "gaze_z",
        "CENTER X": "pupil_center_x",
        "CENTER Y": "pupil_center_y",
    }, inplace=True)
    return df_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video data for gaze estimation.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--data_dir", type=str, default="../raw_data", help="Directory to save the preprocessed data.")
    parser.add_argument("--out_dir", type=str, default="../data", help="Directory to save the preprocessed data.")
    parser.add_argument("--chunksize", type=int, default=30, help="Number of frames per chunk (default: 30s).")
    parser.add_argument("--target_H", type=int, default=120, help="Target height for resized frames.")
    parser.add_argument("--target_W", type=int, default=160, help="Target width for resized frames.")
    args = parser.parse_args()

    dataset_names = ["GazeinTheWild", "Dikablis", "LPW"]
    video_path = args.video_path
    annotation_name = os.path.basename(video_path)
    dataset_name = video_path.split("/")[-3]  # Extract dataset name from the path
    target_H, target_W = args.target_H, args.target_W
    
    assert dataset_name in dataset_names, f"Dataset name {dataset_name} not recognized. Must be one of {dataset_names}."

    annotation_paths = {
        'validity': '{}/{}/ANNOTATIONS/{}validity_pupil.txt'.format(args.data_dir, dataset_name, annotation_name),
        'pupil_center': '{}/{}/ANNOTATIONS/{}pupil_eli.txt'.format(args.data_dir, dataset_name, annotation_name),
        'gaze_vector': '{}/{}/ANNOTATIONS/{}gaze_vec.txt'.format(args.data_dir, dataset_name, annotation_name),
        'iris_mask': '{}/{}/ANNOTATIONS/{}iris_seg_2D.mp4'.format(args.data_dir, dataset_name, annotation_name)
    }

    annotation_df = load_annotations(annotation_paths)
    cap = cv2.VideoCapture(video_path)
    iris_cap = cv2.VideoCapture(annotation_paths['iris_mask'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_iris_frames = int(iris_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert num_frames == annotation_df.shape[0], "Number of frames in video and annotation do not match!"

    chunksize = args.chunksize * fps

    out_video_dir = '{}/{}/{}/VIDEOS'.format(args.out_dir, dataset_name, annotation_name.split(".")[0])
    out_iris_dir = '{}/{}/{}/IRIS'.format(args.out_dir, dataset_name, annotation_name.split(".")[0])
    out_annotation_path = '{}/{}/{}/ANNOTATIONS'.format(args.out_dir, dataset_name, annotation_name.split(".")[0])
    os.makedirs(out_video_dir, exist_ok=True)
    os.makedirs(out_iris_dir, exist_ok=True)
    os.makedirs(out_annotation_path, exist_ok=True)

    video_frames = []
    iris_frames = []
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    assert H / target_H == W / target_W, "Aspect ratio of the video does not match the target aspect ratio!"
    scaling_factor = H / target_H  # or W / target_W, they should be the same due to the assertion above
        
    for _ in tqdm.tqdm(range(num_frames), desc="Loading video frames"):
        ret, frame = cap.read()
        ret_iris, iris_frame = iris_cap.read()
        if not ret_iris or not ret:
            break
        if dataset_name == "GazeinTheWild" and int(video_path[-5])%2 == 1:
            frame = cv2.flip(frame, 0)
        new_W = int(round(frame.shape[1] / scaling_factor))
        new_H = int(round(frame.shape[0] / scaling_factor))
        video_frames.append(cv2.resize(frame, (new_W, new_H)))
        iris_frames.append(cv2.resize(iris_frame, (new_W, new_H)))
    cap.release()
    iris_cap.release()
    video_frames = np.array(video_frames)
    iris_frames = np.array(iris_frames)


    for i in tqdm.tqdm(range(0, num_frames, int(chunksize)), desc="Processing video chunks"):
        out_path = os.path.join(out_video_dir, f"{annotation_name.split('.')[0]}_chunk_{i//int(chunksize)}.mp4")
        if os.path.exists(out_path):
            print(f"Chunk {i//int(chunksize)} already exists, skipping...")
            continue

        video_chunk = video_frames[i:i+int(chunksize)]
        iris_chunk = iris_frames[i:i+int(chunksize)]

        out_iris_path = os.path.join(out_iris_dir, f"{annotation_name.split('.')[0]}_chunk_{i//int(chunksize)}.mp4")
        out_annotation_chunk_path = os.path.join(out_annotation_path, f"{annotation_name.split('.')[0]}_chunk_{i//int(chunksize)}.csv")
        annotation_chunk_df = annotation_df.iloc[i:i+int(chunksize)]
        annotation_chunk_df['pupil_center_x'] = (annotation_chunk_df['pupil_center_x'] / scaling_factor).round().astype(int)
        annotation_chunk_df['pupil_center_y'] = (annotation_chunk_df['pupil_center_y'] / scaling_factor).round().astype(int)
        annotation_chunk_df.to_csv(out_annotation_chunk_path, index=False)

        torchvision.io.write_video(out_path, video_chunk, fps)
        torchvision.io.write_video(out_iris_path, iris_chunk, fps)
    cap.release()
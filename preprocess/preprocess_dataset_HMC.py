import os
import cv2
import tqdm
import glob
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
    parser.add_argument("--data_dir", type=str, default="/data/mtseng/Neurobit/datasets/HMC/data/processed/H04", help="Directory to save the preprocessed data.")
    parser.add_argument("--out_dir", type=str, default="../data", help="Directory to save the preprocessed data.")
    parser.add_argument("--chunksize", type=int, default=30, help="Number of frames per chunk (default: 30s).")
    parser.add_argument("--target_H", type=int, default=120, help="Target height for resized frames.")
    parser.add_argument("--target_W", type=int, default=160, help="Target width for resized frames.")
    args = parser.parse_args()

    dataset_name = 'HMC'
    video_paths = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.mp4"), recursive=True))
    target_H, target_W = args.target_H, args.target_W

    for vid, video_path in enumerate(video_paths):
        print("Processing video: {} ({}/{})".format(video_path, vid+1, len(video_paths)))
        annotation_path = video_path.replace(".mp4", ".csv")
        # assert os.path.exists(video_path), f"Video file {video_path} does not exist!"
        # assert os.path.exists(annotation_path), f"Annotation file {annotation_path} does not exist!"

        if not os.path.exists(video_path) or not os.path.exists(annotation_path):
            print(f"Skipping video {video_path} due to missing video or annotation file.")
            continue

        annotation_df = pd.read_csv(annotation_path)

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        left_eye_frames = []
        right_eye_frames = []

        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        W = W // 2  # Assuming the video has left and right eye frames side by side, we take the width of one eye
        print(f"Original video resolution: {W}x{H}, FPS: {fps}, Total frames: {num_frames}")
        target_H, target_W = 120, 160
        if H / target_H != W / target_W:
            print(f"Warning: Aspect ratio of the video ({W}:{H}) does not match the target aspect ratio ({target_W}:{target_H}). The video will be resized without maintaining the aspect ratio, which may lead to distortion.")
            need_padding = True # pad the y-axis to maintain the aspect ratio
        else:
            need_padding = False

        scaling_factor = W / target_W 
            

        left_annotations = []
        right_annotations = []
        for frame_id in tqdm.tqdm(range(num_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            right_eye_frame = frame[:, :W]  # Assuming the right eye is on the left half of the frame
            left_eye_frame = frame[:, W:]  # Assuming the right eye is on the right half of the frame
            
            # pad the y-axis to maintain the aspect ratio
            if need_padding:
                pad_H = int((scaling_factor * target_H - H) / 2)
                right_eye_frame = cv2.copyMakeBorder(right_eye_frame, pad_H, pad_H, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                left_eye_frame = cv2.copyMakeBorder(left_eye_frame, pad_H, pad_H, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            new_W = int(round(right_eye_frame.shape[1] / scaling_factor))
            new_H = int(round(right_eye_frame.shape[0] / scaling_factor))

            right_eye_frame = cv2.resize(right_eye_frame, (new_W, new_H))
            left_eye_frame = cv2.resize(left_eye_frame, (new_W, new_H))

            right_eye_frames.append(right_eye_frame)
            left_eye_frames.append(left_eye_frame)

            left_pupil_center_x = annotation_df.iloc[frame_id]['left_pupilCenter_x'] - W  # Subtract W to get the coordinate in the left eye frame
            left_pupil_center_y = annotation_df.iloc[frame_id]['left_pupilCenter_y']
            left_pupil_center_x = left_pupil_center_x / scaling_factor
            left_pupil_center_y = (left_pupil_center_y + (pad_H if need_padding else 0)) / scaling_factor
            left_gaze_yaw = annotation_df.iloc[frame_id]['calibrated_OS_pupilCenter_yaw'] * np.pi / 180.0  # Convert from degrees to radians
            left_gaze_pitch = annotation_df.iloc[frame_id]['calibrated_OS_pupilCenter_pitch'] * np.pi / 180.0  # Convert from degrees to radians
            left_gaze_vec = np.stack([
                -np.cos(left_gaze_pitch) * np.sin(left_gaze_yaw),
                -np.sin(left_gaze_pitch),
                np.cos(left_gaze_yaw) * np.cos(left_gaze_pitch)
            ], axis=-1)

            left_annotations.append({
                'FRAME': frame_id + 1,
                'validity': annotation_df.iloc[frame_id]['left_validity'],
                'gaze_x': left_gaze_vec[0],
                'gaze_y': left_gaze_vec[1],
                'gaze_z': left_gaze_vec[2],
                'pupil_center_x': left_pupil_center_x,
                'pupil_center_y': left_pupil_center_y
            })

            right_pupil_center_x = annotation_df.iloc[frame_id]['right_pupilCenter_x']
            right_pupil_center_y = annotation_df.iloc[frame_id]['right_pupilCenter_y']
            right_pupil_center_x = right_pupil_center_x / scaling_factor
            right_pupil_center_y = (right_pupil_center_y + (pad_H if need_padding else 0)) / scaling_factor
            right_gaze_yaw = annotation_df.iloc[frame_id]['calibrated_OD_pupilCenter_yaw'] * np.pi / 180.0  # Convert from degrees to radians
            right_gaze_pitch = annotation_df.iloc[frame_id]['calibrated_OD_pupilCenter_pitch'] * np.pi / 180.0  # Convert from degrees to radians
            right_gaze_vec = np.stack([
                -np.cos(right_gaze_pitch) * np.sin(right_gaze_yaw),
                -np.sin(right_gaze_pitch),
                np.cos(right_gaze_yaw) * np.cos(right_gaze_pitch)
            ], axis=-1)

            right_annotations.append({
                'FRAME': frame_id + 1,
                'validity': annotation_df.iloc[frame_id]['right_validity'],
                'gaze_x': right_gaze_vec[0],
                'gaze_y': right_gaze_vec[1],
                'gaze_z': right_gaze_vec[2],
                'pupil_center_x': right_pupil_center_x,
                'pupil_center_y': right_pupil_center_y
            })
        cap.release()

        left_eye_frames = np.stack(left_eye_frames, axis=0)
        right_eye_frames = np.stack(right_eye_frames, axis=0)
        right_annotations_df = pd.DataFrame(right_annotations)
        left_annotations_df = pd.DataFrame(left_annotations)

        annotation_name = annotation_path.split("/")[-2]
        out_video_dir = '{}/{}/{}/VIDEOS'.format(args.out_dir, dataset_name, annotation_name)
        out_annotation_path = '{}/{}/{}/ANNOTATIONS'.format(args.out_dir, dataset_name, annotation_name)

        os.makedirs(out_video_dir, exist_ok=True)
        os.makedirs(out_annotation_path, exist_ok=True)

        chunksize = args.chunksize * fps
        for i in tqdm.tqdm(range(0, num_frames, int(chunksize))):
            left_out_path = os.path.join(out_video_dir, f"{os.path.basename(video_path).split('.')[0]}_OS_chunk_{i//int(chunksize)}.mp4")
            right_out_path = os.path.join(out_video_dir, f"{os.path.basename(video_path).split('.')[0]}_OD_chunk_{i//int(chunksize)}.mp4")
            if os.path.exists(left_out_path) and os.path.exists(right_out_path):
                print(f"Chunk {i//int(chunksize)} already exists, skipping...")
                continue

            left_video_chunk = left_eye_frames[i:i+int(chunksize)]
            right_video_chunk = right_eye_frames[i:i+int(chunksize)]

            left_out_annotation_chunk_path = left_out_path.replace(out_video_dir, out_annotation_path).replace(".mp4", ".csv")
            right_out_annotation_chunk_path = right_out_path.replace(out_video_dir, out_annotation_path).replace(".mp4", ".csv")

            left_annotation_chunk_df = left_annotations_df.iloc[i:i+int(chunksize)]
            left_annotation_chunk_df.to_csv(left_out_annotation_chunk_path, index=False)

            right_annotation_chunk_df = right_annotations_df.iloc[i:i+int(chunksize)]
            right_annotation_chunk_df.to_csv(right_out_annotation_chunk_path, index=False)

            torchvision.io.write_video(left_out_path, left_video_chunk, fps)
            torchvision.io.write_video(right_out_path, right_video_chunk, fps)
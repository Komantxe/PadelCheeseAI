from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
import argparse

# -------------------------
# Helper functions
# -------------------------
def preprocess_stack(frames, width=640, height=360):
    """Stack 3 consecutive frames into a (1, 9, H, W) input for TrackNet"""
    a = cv2.resize(frames[2], (width, height))
    b = cv2.resize(frames[1], (width, height))
    c = cv2.resize(frames[0], (width, height))
    stacked = np.concatenate((a, b, c), axis=2).astype(np.float32) / 255.0
    stacked = np.transpose(stacked, (2, 0, 1))  # CHW
    stacked = np.expand_dims(stacked, axis=0)   # batch dimension
    return torch.from_numpy(stacked)

def draw_ball(frame, ball_coord, trace=7, prev_coords=[]):
    """Draw ball and its trace on frame"""
    h, w = frame.shape[:2]
    coords_to_draw = prev_coords[-trace:] + [ball_coord]
    for i, coord in enumerate(reversed(coords_to_draw)):
        if coord[0] is not None:
            x = int(coord[0])
            y = int(coord[1])
            cv2.circle(frame, (x, y), radius=0, color=(0,0,255), thickness=max(1, 10-i))
    return frame

def remove_outliers(ball_track, max_dist=100):
    """Remove outlier jumps in detected ball track"""
    arr = np.array([(np.nan if t[0] is None else t[0],
                     np.nan if t[1] is None else t[1]) for t in ball_track], dtype=float)
    dists = np.full(len(arr), -1.0)
    for i in range(1, len(arr)):
        if not np.isnan(arr[i]).any() and not np.isnan(arr[i-1]).any():
            dists[i] = np.linalg.norm(arr[i]-arr[i-1])
    outliers_idx = np.where(dists > max_dist)[0]
    for i in outliers_idx:
        ball_track[i] = (None, None)
    return ball_track

# -------------------------
# Main streaming inference
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--video_out_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--trace', type=int, default=7)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.video_out_path,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (width, height))

    # Keep last 3 frames and track coordinates
    last_frames = []
    ball_track = []

    frame_idx = 0
    print("Processing video...")
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frames.append(frame.copy())
        if len(last_frames) < 3:
            ball_track.append((None,None))
            frame_idx += 1
            pbar.update(1)
            continue

        # Preprocess 3-frame stack
        inp = preprocess_stack(last_frames[-3:]).to(device)

        # Run model
        with torch.no_grad():
            out_pred = model(inp)
            output = out_pred.argmax(dim=1).cpu().numpy()
            x_pred, y_pred = postprocess(output)
            ball_track.append((x_pred, y_pred))

        # Remove temporary memory
        if len(last_frames) > 3:
            last_frames = last_frames[-3:]

        # Draw ball and trace
        frame_out = draw_ball(frame, ball_track[-1], trace=args.trace, prev_coords=ball_track)
        out.write(frame_out)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Remove outliers in full track (optional)
    ball_track = remove_outliers(ball_track)

    print(f"Done! Output video saved to: {args.video_out_path}")

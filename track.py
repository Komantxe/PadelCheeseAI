from pathlib import Path
import time
import sys
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from model import BallTrackerNet
from general import postprocess

SCRIPT_DIR = Path(__file__).parent

VIDEO_PATH = "input.mp4"   # input video here
VIDEO_OUT  = "output_marked.mp4"      # annotated output
MASK_OUT   = "output_mask.mp4"        # mask video output
BALL_MODEL_PATH = SCRIPT_DIR / "model_best.pt" # TrackNet weights
YOLO_MODEL_PATH = SCRIPT_DIR / "yolov8n.pt"    # YOLO weights

# Runtime settings
YOLO_CONF = 0.4
MIN_PLAYER_AREA = 1500
MAX_PLAYERS = 4
MAX_LOST = 30
HIST_COMP_WEIGHT = 0.5
HIST_REASSIGN_THRESH = 0.25
HIST_BINS = [50, 60]
HIST_RANGES = [0, 180, 0, 256]
YOLO_SCALE = 0.75
PROCESS_EVERY = 1
TRACE_LEN = 7
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREVIEW = False   # set True to see preview while running

# helper funcs
def rect_center(r):
    x,y,w,h = r
    return (int(x+w/2), int(y+h/2))

def clamp_rect_to_frame(rect, w, h):
    x, y, ww, hh = rect
    x = int(max(0, min(x, w-1)))
    y = int(max(0, min(y, h-1)))
    ww = int(max(1, min(ww, w-x)))
    hh = int(max(1, min(hh, h-y)))
    return (x, y, ww, hh)

def draw_text(img, text, pt, color=(255,255,255), scale=0.6, thickness=2, bgcolor=(0,0,0)):
    x, y = pt
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(img, (x, y-th-4), (x+tw, y+4), bgcolor, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


class SimpleKalman:
    def __init__(self, dt=1.0, process_noise=1e-2, meas_noise=1e-1):
        self.kf = cv2.KalmanFilter(8, 4, 0)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = dt
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        cv2.setIdentity(self.kf.processNoiseCov, process_noise)
        cv2.setIdentity(self.kf.measurementNoiseCov, meas_noise)
        cv2.setIdentity(self.kf.errorCovPost, 1.0)

    def init(self, bbox):
        x, y, w, h = bbox
        state = np.array([x, y, w, h, 0,0,0,0], dtype=np.float32)
        self.kf.statePost = state.reshape(-1,1)

    def predict(self):
        p = self.kf.predict()
        x, y, w, h = p[0,0], p[1,0], p[2,0], p[3,0]
        return (int(x), int(y), int(abs(w)), int(abs(h)))

    def correct(self, bbox):
        x, y, w, h = bbox
        meas = np.array([x, y, w, h], dtype=np.float32).reshape(-1,1)
        try:
            self.kf.correct(meas)
        except Exception:
            self.init(bbox)
            return bbox
        p = self.kf.statePost
        return (int(p[0,0]), int(p[1,0]), int(abs(p[2,0])), int(abs(p[3,0])))

def compute_hist(frame, bbox):
    x, y, w, h = clamp_rect_to_frame(bbox, frame.shape[1], frame.shape[0])
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return np.zeros((HIST_BINS[0]*HIST_BINS[1],), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist.flatten()

def hist_similarity(h1, h2):
    if h1 is None or h2 is None or h1.size==0 or h2.size==0:
        return 0.0
    val = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
    return (val + 1.0) / 2.0

class PlayerTrack:
    _next_id = 0
    def __init__(self, bbox, frame, label=None):
        self.id = PlayerTrack._next_id
        PlayerTrack._next_id += 1
        self.bbox = bbox
        self.hist = compute_hist(frame, bbox)
        self.kalman = SimpleKalman()
        self.kalman.init(bbox)
        self.time_since_update = 0
        self.label = label
        self.history = deque(maxlen=60)
        self.history.append(rect_center(bbox))
    def predict(self):
        self.bbox = self.kalman.predict()
        self.time_since_update += 1
        self.history.append(rect_center(self.bbox))
        return self.bbox
    def update(self, bbox, frame):
        self.bbox = self.kalman.correct(bbox)
        self.hist = compute_hist(frame, bbox)
        self.time_since_update = 0
        self.history.append(rect_center(self.bbox))

def iou_rect(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0

def match_tracks(tracks, detections, frame):
    if not tracks or not detections:
        return {}, list(range(len(tracks))), list(range(len(detections)))
    t_n = len(tracks); d_n = len(detections)
    scores = np.zeros((t_n, d_n), dtype=np.float32)
    for i, tr in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_s = iou_rect(tr.bbox, det)
            hist_s = hist_similarity(tr.hist, compute_hist(frame, det))
            scores[i,j] = (1.0 - HIST_COMP_WEIGHT) * iou_s + HIST_COMP_WEIGHT * hist_s
    pairs = {}; used_t = set(); used_d = set()
    flat = [(scores[i,j], i, j) for i in range(t_n) for j in range(d_n)]
    flat.sort(reverse=True, key=lambda x: x[0])
    for sc, i, j in flat:
        if sc <= 0: break
        if i in used_t or j in used_d: continue
        if sc < 0.05: continue
        pairs[i] = j; used_t.add(i); used_d.add(j)
    unmatched_tracks = [i for i in range(t_n) if i not in used_t]
    unmatched_dets = [j for j in range(d_n) if j not in used_d]
    return pairs, unmatched_tracks, unmatched_dets

# TrackNet helpers
def preprocess_stack(frames, width=640, height=360):
    a = cv2.resize(frames[2], (width, height))
    b = cv2.resize(frames[1], (width, height))
    c = cv2.resize(frames[0], (width, height))
    stacked = np.concatenate((a, b, c), axis=2).astype(np.float32) / 255.0
    stacked = np.transpose(stacked, (2, 0, 1))
    stacked = np.expand_dims(stacked, axis=0)
    return torch.from_numpy(stacked)

def draw_ball(frame, ball_coord, trace=TRACE_LEN, prev_coords=None):
    if prev_coords is None:
        prev_coords = []
    coords_to_draw = prev_coords[-trace:] + [ball_coord]
    for i, coord in enumerate(reversed(coords_to_draw)):
        if coord is None or coord[0] is None or coord[1] is None:
            continue
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(frame, (x, y), radius=0, color=(0,0,255), thickness=max(1, 10-i))
    return frame

def remove_outliers(ball_track, max_dist=100):
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

# Main runner
def run():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print("ERROR: input video not found:", video_path)
        sys.exit(1)

    ball_model_path = Path(BALL_MODEL_PATH)
    yolo_model_path = Path(YOLO_MODEL_PATH)
    if not ball_model_path.exists():
        print("WARNING: ball model not found at", ball_model_path, " — script may fail.")
    if not yolo_model_path.exists():
        print("WARNING: yolo model not found at", yolo_model_path, " — script may fail.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading TrackNet model...", end=" ", flush=True)
    ball_model = BallTrackerNet()
    try:
        ball_state = torch.load(str(ball_model_path), map_location=device)
        ball_model.load_state_dict(ball_state)
    except Exception as e:
        print("\nERROR loading TrackNet model:", e)
        raise
    ball_model.to(device).eval()
    print("done")

    print("Loading YOLO model...", yolo_model_path)
    yolo = YOLO(str(yolo_model_path))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_marked = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (W, H))
    out_mask = cv2.VideoWriter(str(MASK_OUT), fourcc, fps, (W, H), isColor=False)

    player_tracks = []
    player_label_assigned = False
    label_hist = {}

    last_frames = []
    ball_track = []

    pbar = tqdm(total=frame_count, desc="Processing frames")
    frame_idx = -1
    last_show_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if PROCESS_EVERY > 1 and (frame_idx % PROCESS_EVERY) != 0:
            pbar.update(1)
            continue

        orig = frame.copy()

        # TrackNet (ball)
        last_frames.append(frame.copy())
        if len(last_frames) < 3:
            ball_track.append((None, None))
        else:
            inp = preprocess_stack(last_frames[-3:]).to(device)
            with torch.no_grad():
                out_pred = ball_model(inp)
                output = out_pred.argmax(dim=1).cpu().numpy()
                x_pred, y_pred = postprocess(output)
                ball_track.append((x_pred, y_pred))
            if len(last_frames) > 3:
                last_frames = last_frames[-3:]

        # YOLO (players)
        small = cv2.resize(frame, (0,0), fx=YOLO_SCALE, fy=YOLO_SCALE) if YOLO_SCALE != 1.0 else frame
        results = yolo.predict(small, imgsz=int(640*YOLO_SCALE), conf=YOLO_CONF, classes=[0], verbose=False)
        dets = []
        if len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes'):
                for box in r.boxes:
                    try:
                        xy = box.xyxy[0].tolist()
                    except Exception:
                        xy = box.xyxy.tolist()[0]
                    x1,y1,x2,y2 = xy
                    scale_back = 1.0 / YOLO_SCALE
                    x1 = int(round(x1 * scale_back)); y1 = int(round(y1 * scale_back))
                    x2 = int(round(x2 * scale_back)); y2 = int(round(y2 * scale_back))
                    x, y = int(x1), int(y1)
                    w_box, h_box = int(x2-x1), int(y2-y1)
                    if w_box * h_box < MIN_PLAYER_AREA:
                        continue
                    dets.append((x, y, w_box, h_box))

        for t in player_tracks:
            t.predict()

        pairs, unmatched_tracks, unmatched_dets = match_tracks(player_tracks, dets, frame)
        for t_idx, d_idx in pairs.items():
            player_tracks[t_idx].update(dets[d_idx], frame)

        for d_idx in unmatched_dets:
            if len(player_tracks) < MAX_PLAYERS:
                player_tracks.append(PlayerTrack(dets[d_idx], frame))

        if not player_label_assigned and len(player_tracks) >= MAX_PLAYERS:
            stable_tracks = sorted(player_tracks, key=lambda t: -len(t.history))[:MAX_PLAYERS]
            stable_tracks = sorted(stable_tracks, key=lambda t: rect_center(t.bbox)[0])
            for i, tr in enumerate(stable_tracks):
                tr.label = f"player{i+1}"
                label_hist[tr.label] = tr.hist.copy()
            player_label_assigned = True

        if player_label_assigned:
            for t in player_tracks:
                if t.label:
                    label_hist[t.label] = t.hist.copy()
            current_labels = {t.label for t in player_tracks if t.label}
            all_labels = {f"player{i+1}" for i in range(MAX_PLAYERS)}
            missing_labels = list(all_labels - current_labels)
            unlabeled_tracks = [t for t in player_tracks if not t.label]
            if missing_labels and unlabeled_tracks:
                for label in missing_labels:
                    if label not in label_hist:
                        continue
                    best_score = -1.0; best_tr = None
                    for tr in unlabeled_tracks:
                        s = hist_similarity(label_hist[label], tr.hist)
                        if s > best_score:
                            best_score = s; best_tr = tr
                    if best_tr is not None and best_score >= HIST_REASSIGN_THRESH:
                        best_tr.label = label
                        label_hist[label] = best_tr.hist.copy()
                        unlabeled_tracks.remove(best_tr)

        player_tracks = [t for t in player_tracks if t.time_since_update <= MAX_LOST]

        # visualization
        vis = orig.copy()
        mask_frame = np.zeros((H, W), dtype=np.uint8)

        for tr in player_tracks:
            x,y,ww,hh = clamp_rect_to_frame(tr.bbox, W, H)
            color = (0,128,255) if tr.label else (100,100,255)
            cv2.rectangle(vis, (x,y), (x+ww, y+hh), color, 2)
            draw_text(vis, tr.label if tr.label else f"ID{tr.id}", (x, y-6), color=(255,255,255), bgcolor=color)
            cx,cy = rect_center((x,y,ww,hh))
            cv2.circle(mask_frame, (cx,cy), max(10, int(max(ww,hh)/3)), 255, -1)
            for i in range(1, len(tr.history)):
                cv2.line(vis, tr.history[i-1], tr.history[i], (200,200,200), 1)

        current_ball = ball_track[-1] if len(ball_track)>0 else (None, None)
        vis = draw_ball(vis, current_ball, trace=TRACE_LEN, prev_coords=ball_track)
        if current_ball[0] is not None and current_ball[1] is not None:
            cv2.circle(mask_frame, (int(current_ball[0]), int(current_ball[1])), 5, 255, -1)

        out_marked.write(vis)
        out_mask.write(mask_frame)

        if PREVIEW and (time.time()-last_show_time > 0.03):
            cv2.imshow("vis", vis)
            cv2.imshow("mask", mask_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_show_time = time.time()

        pbar.update(1)

    pbar.close()
    cap.release(); out_marked.release(); out_mask.release()
    cv2.destroyAllWindows()

    print("Output saved:")
    print("  annotated video:", VIDEO_OUT)
    print("  mask video     :", MASK_OUT)

if __name__ == "__main__":
    run()

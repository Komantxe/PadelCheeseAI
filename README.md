# Ball + Player Tracking

This project runs **ball tracking (TrackNet)** and **player tracking
(YOLOv8 + Kalman filter)** together on a video.\
It produces two outputs: - `output_marked.mp4` → video with players
and ball marked - `output_mask.mp4` → mask video (players & ball in
white on black)-

## How to Run

1.  **Clone this repo** and move into the folder:

``` bash
git clone https://github.com/Komantxe/PadelCheeseAI.git
cd PadelCheeseAI
```

2.  **Create a virtual environment**

``` bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3.  **Install dependencies**:

``` bash
pip install -r requirements.txt
```

4.  **Edit the script to set your input/output video paths**:\
    Open `track.py` and update:

``` python
VIDEO_PATH = Path("videos/input.mp4")
VIDEO_OUT  = Path("videos/output_marked.mp4")
MASK_OUT   = Path("videos/output_mask.mp4")
```

6.  **Run the script**:

``` bash
python track.py
```

------------------------------------------------------------------------

## ✅ Output

-   Annotated video → `combined_marked.mp4`
-   Mask video → `combined_mask.mp4`

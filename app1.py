# # app.py
# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# from collections import OrderedDict, deque

# # ---------------------------
# # Helper classes and methods
# # ---------------------------

# class CentroidTracker:
#     """
#     Simple centroid tracker: assigns incremental IDs to detections and matches by centroid distance.
#     Not as robust as SORT/DeepSORT but lightweight and works well for simple scenes.
#     """
#     def __init__(self, max_disappeared=50, max_distance=50):
#         self.nextObjectID = 0
#         self.objects = OrderedDict()  # objectID -> centroid
#         self.disappeared = OrderedDict()  # objectID -> frames disappeared
#         self.max_disappeared = max_disappeared
#         self.max_distance = max_distance

#     def register(self, centroid):
#         self.objects[self.nextObjectID] = centroid
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1

#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]

#     def update(self, rects):
#         # rects: list of bounding boxes (startX, startY, endX, endY)
#         if len(rects) == 0:
#             # mark all as disappeared
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)
#             return self.objects

#         input_centroids = np.zeros((len(rects), 2), dtype="int")
#         for (i, (startX, startY, endX, endY)) in enumerate(rects):
#             cX = int((startX + endX) / 2.0)
#             cY = int((startY + endY) / 2.0)
#             input_centroids[i] = (cX, cY)

#         if len(self.objects) == 0:
#             for i in range(0, len(input_centroids)):
#                 self.register(input_centroids[i])
#         else:
#             objectIDs = list(self.objects.keys())
#             objectCentroids = list(self.objects.values())

#             # compute distance matrix between existing and new centroids
#             D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

#             rows = D.min(axis=1).argsort()
#             cols = D.argmin(axis=1)[rows]

#             usedRows, usedCols = set(), set()
#             for (row, col) in zip(rows, cols):
#                 if row in usedRows or col in usedCols:
#                     continue
#                 if D[row, col] > self.max_distance:
#                     continue
#                 objectID = objectIDs[row]
#                 self.objects[objectID] = input_centroids[col]
#                 self.disappeared[objectID] = 0
#                 usedRows.add(row)
#                 usedCols.add(col)

#             unusedRows = set(range(0, D.shape[0])).difference(usedRows)
#             unusedCols = set(range(0, D.shape[1])).difference(usedCols)

#             # mark disappeared
#             for row in unusedRows:
#                 objectID = objectIDs[row]
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)

#             # register new ones
#             for col in unusedCols:
#                 self.register(input_centroids[col])

#         return self.objects

# # For counting: keep track per ID of past centroid positions
# class TrackableObject:
#     def __init__(self, objectID, centroid):
#         self.objectID = objectID
#         self.centroids = deque([centroid], maxlen=30)
#         self.counted = False

# def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
#     # brightness: -127..127, contrast: -127..127
#     if brightness != 0:
#         if brightness > 0:
#             shadow = brightness
#             highlight = 255
#         else:
#             shadow = 0
#             highlight = 255 + brightness
#         alpha_b = (highlight - shadow) / 255
#         gamma_b = shadow
#         buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
#     else:
#         buf = input_img.copy()

#     if contrast != 0:
#         f = 131*(contrast + 127)/(127*(131-contrast))
#         alpha_c = f
#         gamma_c = 127*(1-f)
#         buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

#     return buf

# def estimate_distance(focal_length, real_width, pixel_width):
#     # focal_length and real_width must be in same unit (e.g. cm)
#     # pixel_width is width in pixels
#     if pixel_width <= 0 or focal_length <= 0:
#         return None
#     return (real_width * focal_length) / pixel_width

# # ---------------------------
# # Streamlit UI + main logic
# # ---------------------------

# st.set_page_config(page_title="YOLOv8 + OpenCV Multi-task App", layout="wide")
# st.title("YOLOv8 + OpenCV — Combined Multi-task Streamlit App 🚀")

# # Sidebar controls
# st.sidebar.header("Model & Input")
# model_path = st.sidebar.text_input("YOLO model path (local)", "yolov8n.pt")
# use_webcam = st.sidebar.checkbox("Use local webcam (VideoCapture)", value=False)
# uploaded_file = st.sidebar.file_uploader("Or upload a video file (mp4/mov/avi)", type=["mp4","mov","avi"])
# conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.01)
# iou_thresh = st.sidebar.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
# resize_width = st.sidebar.number_input("Resize frames width (px, 0 = original)", min_value=0, value=800)
# max_disappeared = st.sidebar.number_input("Tracker max disappeared frames", min_value=1, value=40)
# max_distance = st.sidebar.number_input("Tracker max centroid matching distance (px)", min_value=1, value=80)
# st.sidebar.markdown("---")
# st.sidebar.header("Counting & Distance")
# line_orientation = st.sidebar.selectbox("Counting line orientation", ["horizontal", "vertical"])
# line_position = st.sidebar.slider("Line position (% from top/left)", 0, 100, 50)
# count_direction = st.sidebar.selectbox("Count direction to increase count", ["positive", "both"])
# st.sidebar.markdown("---")
# st.sidebar.header("Preprocess")
# brightness = st.sidebar.slider("Brightness (-127..127)", -50, 50, 0)
# contrast = st.sidebar.slider("Contrast (-127..127)", -30, 30, 0)
# st.sidebar.markdown("---")
# st.sidebar.header("Distance estimation (optional)")
# real_object_width = st.sidebar.number_input("Real object width (cm) for distance estimate (0 to disable)", min_value=0.0, value=0.0)
# focal_length = st.sidebar.number_input("Focal length (px) — calibrate or approximate (0 to disable)", min_value=0.0, value=0.0)
# st.sidebar.markdown("To estimate focal length: focal = (pixel_width * known_distance) / real_width")

# st.sidebar.markdown("---")
# run_button = st.sidebar.button("Start Processing")
# stop_button = st.sidebar.button("Stop")

# # Main area
# col1, col2 = st.columns([2,1])
# frame_display = col1.empty()
# stats_area = col2.empty()

# # Load model
# @st.cache_resource(show_spinner=False)
# def load_yolo_model(path):
#     try:
#         model = YOLO(path)
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None

# model = load_yolo_model(model_path)

# # State containers
# if "tracking" not in st.session_state:
#     st.session_state.tracking = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
# if "trackable_objects" not in st.session_state:
#     st.session_state.trackable_objects = {}
# if "counts" not in st.session_state:
#     st.session_state.counts = 0
# if "processing" not in st.session_state:
#     st.session_state.processing = False

# # Update tracker parameters live if sidebar changes
# st.session_state.tracking.max_disappeared = max_disappeared
# st.session_state.tracking.max_distance = max_distance

# def process_stream(source_capture):
#     fps_time = time.time()
#     frame_count = 0
#     last_display_time = time.time()
#     while source_capture.isOpened() and st.session_state.processing:
#         grabbed, frame = source_capture.read()
#         if not grabbed:
#             break

#         frame_count += 1
#         # resize if requested
#         if resize_width > 0:
#             h, w = frame.shape[:2]
#             scale = resize_width / float(w)
#             frame = cv2.resize(frame, (resize_width, int(h * scale)))

#         # apply brightness/contrast
#         proc_frame = apply_brightness_contrast(frame, brightness, contrast)

#         # YOLO expects RGB
#         rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

#         # Run YOLO model (single frame inference)
#         try:
#             results = model.predict(rgb, imgsz=640, device='cpu', conf=conf_thresh, iou=iou_thresh, verbose=False)
#         except Exception as e:
#             st.error(f"YOLO inference failed: {e}")
#             st.session_state.processing = False
#             break

#         boxes = []
#         confidences = []
#         class_ids = []
#         names = model.names if hasattr(model, "names") else {}

#         if len(results) > 0:
#             r = results[0]
#             if hasattr(r, 'boxes'):
#                 for box in r.boxes:
#                     xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # x1,y1,x2,y2
#                     conf = float(box.conf.cpu().numpy()[0])
#                     cls = int(box.cls.cpu().numpy()[0])
#                     if conf < conf_thresh:
#                         continue
#                     x1, y1, x2, y2 = xyxy.tolist()
#                     boxes.append((x1, y1, x2, y2))
#                     confidences.append(conf)
#                     class_ids.append(cls)

#         # Update centroid tracker
#         objects = st.session_state.tracking.update(boxes)

#         # Update trackable objects and counting
#         for objectID, centroid in objects.items():
#             to = st.session_state.trackable_objects.get(objectID, None)
#             if to is None:
#                 to = TrackableObject(objectID, centroid)
#             else:
#                 to.centroids.append(centroid)
#             st.session_state.trackable_objects[objectID] = to

#             # counting logic: check if object's centroid crossed the line
#             if not to.counted:
#                 if line_orientation == "horizontal":
#                     y_coords = [c[1] for c in to.centroids]
#                     if len(y_coords) >= 2:
#                         # check crossing relative to line_position
#                         line_y = int((line_position / 100.0) * proc_frame.shape[0])
#                         if count_direction == "positive":
#                             # for positive direction: last y < line <= previous y or vice versa depending on flow
#                             if y_coords[-2] < line_y <= y_coords[-1]:
#                                 st.session_state.counts += 1
#                                 to.counted = True
#                         else:
#                             # both directions
#                             if (y_coords[-2] < line_y <= y_coords[-1]) or (y_coords[-2] > line_y >= y_coords[-1]):
#                                 st.session_state.counts += 1
#                                 to.counted = True
#                 else:  # vertical
#                     x_coords = [c[0] for c in to.centroids]
#                     if len(x_coords) >= 2:
#                         line_x = int((line_position / 100.0) * proc_frame.shape[1])
#                         if count_direction == "positive":
#                             if x_coords[-2] < line_x <= x_coords[-1]:
#                                 st.session_state.counts += 1
#                                 to.counted = True
#                         else:
#                             if (x_coords[-2] < line_x <= x_coords[-1]) or (x_coords[-2] > line_x >= x_coords[-1]):
#                                 st.session_state.counts += 1
#                                 to.counted = True

#         # Draw boxes and IDs
#         out = proc_frame.copy()
#         for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_ids)):
#             x1, y1, x2, y2 = box
#             label = f"{names.get(cls, cls)} {conf:.2f}"
#             cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(out, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#             # distance estimate if enabled
#             if real_object_width > 0 and focal_length > 0:
#                 pixel_w = x2 - x1
#                 dist = estimate_distance(focal_length, real_object_width, pixel_w)
#                 if dist is not None:
#                     txt = f"{dist:.1f}cm"
#                     cv2.putText(out, txt, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

#         # Draw object IDs
#         for objectID, centroid in objects.items():
#             text = f"ID {objectID}"
#             cv2.putText(out, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#             cv2.circle(out, (centroid[0], centroid[1]), 4, (0,0,255), -1)

#         # draw counting line
#         if line_orientation == "horizontal":
#             y_line = int((line_position / 100.0) * out.shape[0])
#             cv2.line(out, (0, y_line), (out.shape[1], y_line), (255,0,0), 2)
#         else:
#             x_line = int((line_position / 100.0) * out.shape[1])
#             cv2.line(out, (x_line, 0), (x_line, out.shape[0]), (255,0,0), 2)

#         # overlay counts and FPS
#         cv2.putText(out, f"Counts: {st.session_state.counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
#         fps = 1.0 / (time.time() - fps_time) if (time.time() - fps_time) > 0 else 0.0
#         fps_time = time.time()
#         cv2.putText(out, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#         # display
#         frame_display.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")

#         # small break to allow Streamlit to update buttons etc
#         if stop_button:
#             st.session_state.processing = False
#             break

#     source_capture.release()

# # Start / stop logic
# if run_button:
#     if model is None:
#         st.error("Model failed to load. Check model path.")
#     else:
#         st.session_state.processing = True
#         st.session_state.counts = 0
#         st.session_state.trackable_objects = {}
#         st.session_state.tracking = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)

#         if use_webcam:
#             cap = cv2.VideoCapture(0)
#             if not cap.isOpened():
#                 st.error("Cannot open webcam (VideoCapture(0)). Make sure you run Streamlit locally and grant camera permissions.")
#                 st.session_state.processing = False
#             else:
#                 process_stream(cap)
#         elif uploaded_file is not None:
#             tfile = uploaded_file
#             # save to temp file
#             tpath = f"temp_{int(time.time())}.mp4"
#             with open(tpath, "wb") as f:
#                 f.write(tfile.read())
#             cap = cv2.VideoCapture(tpath)
#             if not cap.isOpened():
#                 st.error("Failed to open uploaded video.")
#                 st.session_state.processing = False
#             else:
#                 process_stream(cap)
#         else:
#             st.info("No input selected. Upload a video or enable webcam in sidebar.")

# if stop_button:
#     st.session_state.processing = False
#     st.success("Stopped processing.")

# # Show stats and small help
# stats_area.markdown(
#     f"""
# **Model:** `{model_path}`  
# **Processing:** {'Running' if st.session_state.processing else 'Stopped'}  
# **Total counted:** **{st.session_state.counts}**  

# **Notes:**  
# - This app uses a simple centroid tracker. For crowded scenes consider SORT / DeepSORT.  
# - For distance estimates you must provide `real object width (cm)` and `focal length (px)`.  
#   Example: measure pixel width of a known object at known distance to compute focal length.  
# - If using webcam locally, run `streamlit run app.py` on your local machine.
# """
# )


# app.py
import streamlit as st
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from collections import OrderedDict, deque

# -------------------------
# Helper utils & tracker
# -------------------------
st.set_page_config(page_title="YOLO Multi-model + OpenCV Detector", layout="wide")

def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"

def normalize_names(model):
    if model is None:
        return {}
    try:
        names = model.names
    except Exception:
        try:
            names = model.model.names
        except Exception:
            names = {}
    if isinstance(names, (list, tuple)):
        return {i: n for i, n in enumerate(names)}
    return dict(names)

def safe_box_extract(box):
    # robustly extract xyxy, conf, cls from ultralytics Box object
    try:
        xy = box.xyxy.cpu().numpy().ravel()[:4].astype(int)
    except Exception:
        xy = np.array(box.xyxy).ravel()[:4].astype(int)
    x1,y1,x2,y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
    try:
        conf = float(box.conf.cpu().item())
    except Exception:
        conf = float(box.conf)
    try:
        cls = int(box.cls.cpu().item())
    except Exception:
        cls = int(box.cls)
    return x1,y1,x2,y2,cls,conf

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=80):
        self.nextID = 0
        self.objects = OrderedDict()   # id -> centroid
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, oid):
        if oid in self.objects: del self.objects[oid]
        if oid in self.disappeared: del self.disappeared[oid]

    def update(self, rects):
        # rects: list of (x1,y1,x2,y2)
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1,y1,x2,y2) in enumerate(rects):
            input_centroids[i] = (int((x1+x2)/2), int((y1+y2)/2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (r, c) in zip(rows, cols):
                if r in usedRows or c in usedCols: continue
                if D[r, c] > self.max_distance: continue
                oid = objectIDs[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.disappeared[oid] = 0
                usedRows.add(r); usedCols.add(c)

            unusedRows = set(range(D.shape[0])) - usedRows
            for r in unusedRows:
                oid = objectIDs[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            unusedCols = set(range(D.shape[1])) - usedCols
            for c in unusedCols:
                self.register(tuple(input_centroids[c]))

        return self.objects

class TrackableObject:
    def __init__(self, oid, centroid, cls=None):
        self.oid = oid
        self.centroids = deque([centroid], maxlen=30)
        self.counted = False
        self.cls = cls  # assigned class id for this object

# -------------------------
# UI: model choice + options
# -------------------------
st.title("YOLO Multi-model Detector — Detect & Count (Birds,insects, snakes, helmets, cars, etc.)")

# model options & descriptions
MODEL_OPTIONS = {
    "yolo12n.pt": "YOLOv12-n (very fast) — good for low-power devices, 480p–1080p scanning",
    "yolo11n.pt": "YOLOv11-n (fast) — balanced for field use (480p–1080p)",
    "yolo11s.pt": "YOLOv11-s (balanced) — better accuracy for snakes/insects (720p+)",
    "yolov8n.pt": "YOLOv8-n (fallback) — default small Ultralytics model",
    "yolov8s.pt": "YOLOv8-s — higher accuracy than n, slower",
    "custom": "Custom model — provide full path to your .pt"
}

model_choice = st.sidebar.selectbox("Choose model", list(MODEL_OPTIONS.keys()), format_func=lambda x: f"{x} — {MODEL_OPTIONS[x]}")
custom_model_path = ""
if model_choice == "custom":
    custom_model_path = st.sidebar.text_input("Path to custom .pt (local or /content/...)", "")

# input source
st.sidebar.header("Input")
use_webcam = st.sidebar.checkbox("Use webcam (local only)", value=False)
uploaded = st.sidebar.file_uploader("Upload video (mp4/mov/avi)", type=["mp4","mov","avi"])
stream_url = st.sidebar.text_input("Or RTSP/HTTP stream URL (optional)", "")

# detection params
st.sidebar.header("Detection & Perf")
conf_thresh = st.sidebar.slider("Confidence", 0.0, 1.0, 0.35, 0.01)
iou_thresh = st.sidebar.slider("IoU NMS", 0.0, 1.0, 0.45, 0.01)
resize_width = st.sidebar.number_input("Resize width (px, 0 = original)", min_value=0, value=800)
frame_skip = st.sidebar.number_input("Process every Nth frame (skip)", min_value=1, value=1)
max_disp = st.sidebar.number_input("Tracker max disappeared frames", min_value=1, value=40)
max_dist = st.sidebar.number_input("Tracker max centroid distance (px)", min_value=10, value=80)
debug_raw = st.sidebar.checkbox("Show raw predictions (debug)", value=False)

# preprocess & distance
st.sidebar.header("Preprocess & Distance")
brightness = st.sidebar.slider("Brightness (-127..127)", -50, 50, 0)
contrast = st.sidebar.slider("Contrast (-127..127)", -30, 30, 0)
real_width = st.sidebar.number_input("Real object width (cm) for distance (0 disable)", min_value=0.0, value=0.0)
focal_px = st.sidebar.number_input("Focal length (px) (0 disable)", min_value=0.0, value=0.0)

# control
run_btn = st.sidebar.button("Start")
stop_btn = st.sidebar.button("Stop")

# main layout
col1, col2 = st.columns([2,1])
frame_area = col1.empty()
info_area = col2

# -------------------------
# Model loader (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        m = YOLO(path)
        return m
    except Exception as e:
        return None

# choose model path
if model_choice == "custom":
    model_path = custom_model_path.strip()
else:
    model_path = model_choice

model = None
if model_path:
    model = load_model(model_path)
    if model is None:
        st.sidebar.error(f"Failed to load model at: {model_path}")
    else:
        st.sidebar.success(f"Loaded model: {model_path}")

names = normalize_names(model)

# session state initialization
if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker(max_disappeared=max_disp, max_distance=max_dist)
if "tracks" not in st.session_state:
    st.session_state.tracks = {}  # oid -> TrackableObject
if "counts" not in st.session_state:
    st.session_state.counts = {}  # class_name -> int
if "processing" not in st.session_state:
    st.session_state.processing = False

# update tracker params live
st.session_state.tracker.max_disappeared = max_disp
st.session_state.tracker.max_distance = max_dist

# helpers
def apply_brightness_contrast(img, b=0, c=0):
    if b != 0:
        if b > 0:
            shadow = b; highlight = 255
        else:
            shadow = 0; highlight = 255 + b
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    if c != 0:
        f = 131*(c + 127)/(127*(131-c))
        alpha_c = f; gamma_c = 127*(1-f)
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    return img

def estimate_distance(focal, real_w, pix_w):
    if pix_w <= 0 or focal <= 0: return None
    return (real_w * focal) / pix_w

device = device_name()

# processing loop
def process_capture(cap):
    fps_time = time.time()
    frame_idx = 0

    # reset counters/tracks
    st.session_state.counts = {}
    st.session_state.tracks = {}
    st.session_state.tracker = CentroidTracker(max_disappeared=max_disp, max_distance=max_dist)

    while cap.isOpened() and st.session_state.processing:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_skip > 1 and (frame_idx % frame_skip) != 0:
            continue

        # resize
        if resize_width and resize_width > 0:
            h,w = frame.shape[:2]
            scale = resize_width / float(w)
            frame = cv2.resize(frame, (resize_width, int(h*scale)))

        proc = apply_brightness_contrast(frame, brightness, contrast)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        # inference
        try:
            results = model.predict(rgb, imgsz=640, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.session_state.processing = False
            break

        boxes = []
        cls_ids = []
        confs = []
        raw_preds = []

        if len(results) > 0:
            r = results[0]
            for box in getattr(r, "boxes", []):
                x1,y1,x2,y2,cls,conf = safe_box_extract(box)
                if conf < conf_thresh: 
                    continue
                boxes.append((x1,y1,x2,y2))
                cls_ids.append(cls)
                confs.append(conf)
                raw_preds.append((cls, names.get(cls, str(cls)), conf))

        # debug raw preds
        if debug_raw and raw_preds:
            st.sidebar.write("Raw preds (cls_id, name, conf):")
            st.sidebar.write(raw_preds[:8])

        # update tracker (centroids)
        objects = st.session_state.tracker.update(boxes)

        # for each tracked ID, associate nearest detection box (so track keeps a class)
        for oid, centroid in objects.items():
            # find nearest box index
            assigned_idx = None
            min_dist = float("inf")
            for i, b in enumerate(boxes):
                bx = int((b[0]+b[2]) / 2); by = int((b[1]+b[3]) / 2)
                d = (bx - centroid[0])**2 + (by - centroid[1])**2
                if d < min_dist:
                    min_dist = d; assigned_idx = i
            to = st.session_state.tracks.get(oid, None)
            if to is None:
                # if assigned_idx exists, set class, else None
                cls_for_id = cls_ids[assigned_idx] if assigned_idx is not None else None
                to = TrackableObject(oid, centroid)
                to.cls = cls_for_id
                st.session_state.tracks[oid] = to
            else:
                to.centroids.append(centroid)
                # update class if we find one now and didn't have earlier
                if to.cls is None and assigned_idx is not None:
                    to.cls = cls_ids[assigned_idx]

            # counting: check crossing line and increment class-specific count
            if not to.counted and to.cls is not None:
                cls_name = names.get(to.cls, str(to.cls))
                if line_orientation == "horizontal":
                    y_coords = [c[1] for c in to.centroids]
                    if len(y_coords) >= 2:
                        line_y = int((line_position / 100.0) * proc.shape[0])
                        if count_direction == "positive":
                            if y_coords[-2] < line_y <= y_coords[-1]:
                                st.session_state.counts[cls_name] = st.session_state.counts.get(cls_name, 0) + 1
                                to.counted = True
                        else:
                            if (y_coords[-2] < line_y <= y_coords[-1]) or (y_coords[-2] > line_y >= y_coords[-1]):
                                st.session_state.counts[cls_name] = st.session_state.counts.get(cls_name, 0) + 1
                                to.counted = True
                else:
                    x_coords = [c[0] for c in to.centroids]
                    if len(x_coords) >= 2:
                        line_x = int((line_position / 100.0) * proc.shape[1])
                        if count_direction == "positive":
                            if x_coords[-2] < line_x <= x_coords[-1]:
                                st.session_state.counts[cls_name] = st.session_state.counts.get(cls_name, 0) + 1
                                to.counted = True
                        else:
                            if (x_coords[-2] < line_x <= x_coords[-1]) or (x_coords[-2] > line_x >= x_coords[-1]):
                                st.session_state.counts[cls_name] = st.session_state.counts.get(cls_name, 0) + 1
                                to.counted = True

        # draw boxes, labels, IDs
        out = proc.copy()
        for (b, cls_id, conf) in zip(boxes, cls_ids, confs):
            x1,y1,x2,y2 = b
            name = names.get(cls_id, str(cls_id))
            lbl = f"{name} {conf:.2f}"
            cv2.rectangle(out, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.putText(out, lbl, (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # distance if configured
            if real_width > 0 and focal_px > 0:
                pix_w = x2 - x1
                d = estimate_distance(focal_px, real_width, pix_w)
                if d is not None:
                    cv2.putText(out, f"{d:.1f}cm", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        for oid, centroid in objects.items():
            cv2.putText(out, f"ID {oid}", (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.circle(out, (centroid[0], centroid[1]), 4, (0,0,255), -1)

        # draw counting line
        if line_orientation == "horizontal":
            ly = int((line_position / 100.0) * out.shape[0])
            cv2.line(out, (0,ly), (out.shape[1], ly), (255,0,0), 2)
        else:
            lx = int((line_position / 100.0) * out.shape[1])
            cv2.line(out, (lx,0), (lx,out.shape[0]), (255,0,0), 2)

        # overlay FPS & counts
        fps = 1.0 / (time.time() - fps_time) if (time.time() - fps_time) > 0 else 0.0
        fps_time = time.time()
        cv2.putText(out, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # display frame in streamlit (use_container_width -> no deprecation warnings)
        frame_area.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # stop button check
        if stop_btn:
            st.session_state.processing = False
            break

    cap.release()
    
# Choose line orientation: 'horizontal' or 'vertical'
line_orientation = "horizontal"
line_position = 50  # default line position (50%)
# Choose count direction: 'up' or 'down'
count_direction = "up"



# -------------------------
# Start / Stop logic
# -------------------------
if run_btn:
    if model is None:
        st.error("Model not loaded. Check path or pick a model available in working directory.")
    else:
        st.session_state.processing = True
        # choose source
        if use_webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to open local webcam. Run locally & allow camera.")
                st.session_state.processing = False
            else:
                process_capture(cap)
        elif uploaded is not None:
            tmp_path = f"temp_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f: f.write(uploaded.read())
            cap = cv2.VideoCapture(tmp_path)
            process_capture(cap)
            try: os.remove(tmp_path)
            except: pass
        elif stream_url:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                st.error("Failed to open stream URL.")
                st.session_state.processing = False
            else:
                process_capture(cap)
        else:
            st.info("Select webcam, upload video, or paste stream URL and press Start.")

if stop_btn:
    st.session_state.processing = False
    st.success("Stopped processing.")

# -------------------------
# Stats & counts display
# -------------------------
info_text = f"**Model:** `{model_path if 'model_path' in locals() else model_choice}`  \n"
info_text += f"**Device:** {device}  \n"
info_text += f"**Processing:** {'Running' if st.session_state.processing else 'Stopped'}  \n\n"
info_text += "**Counts (by class):**  \n"
if st.session_state.counts:
    for k,v in st.session_state.counts.items():
        info_text += f"- {k}: {v}  \n"
else:
    info_text += "- (no counts yet)\n"

info_area.markdown(info_text)

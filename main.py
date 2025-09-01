from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")

# Define class weights (tune these as per your city traffic conditions)
weights = {
    "car": 1,
    "motorcycle": 0.5,
    "bus": 3,
    "truck": 3,
}

# Default signal time per side
BASE_TIME = 30
MIN_TIME = 10
MAX_TIME = 90

# Example: 4 images = 4 sides of an intersection
image_paths = ["test_images/imtest1.jpeg", "test_images/imtest2.jpeg", "test_images/imtest3.jpeg", "test_images/imtest4.jpeg"]

# Run YOLO on all sides
results = model(image_paths)

traffic_loads = []
for i, result in enumerate(results):
    counts = {}
    total_load = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in weights:
            counts[cls_name] = counts.get(cls_name, 0) + 1
            total_load += weights[cls_name]
    traffic_loads.append(total_load)
    print(f"Side {i+1} load breakdown:", counts, " Total Load =", total_load)

# Normalize loads to distribute time
total_load_all = sum(traffic_loads)
signal_times = []

for load in traffic_loads:
    if total_load_all == 0:
        time = BASE_TIME  # No traffic -> default
    else:
        time = int((load / total_load_all) * (BASE_TIME * 4))
        time = max(MIN_TIME, min(MAX_TIME, time))
    signal_times.append(time)

print("\nRecommended Signal Times (in seconds):")
for i, t in enumerate(signal_times):
    print(f"Side {i+1}: {t}s")

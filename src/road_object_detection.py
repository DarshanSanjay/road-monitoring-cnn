from pathlib import Path

import cv2
from ultralytics import YOLO


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# Load the pretrained YOLO CNN model
model = YOLO("yolo11n.pt")


# Objects relevant to road monitoring
ROAD_OBJECTS = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


def detect_objects(image_path):
    """Detect road-related objects in an image."""
    
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    results = model.predict(
        source=image,
        conf=0.35,
        verbose=False
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name not in ROAD_OBJECTS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detection = {
            "object": class_name,
            "confidence": round(confidence, 3),
            "bounding_box": [
                round(x1),
                round(y1),
                round(x2),
                round(y2),
            ],
        }

        detections.append(detection)

    return image, detections


def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image."""
    
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bounding_box"]

        label = (
            f'{detection["object"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output


def process_image(image_name):
    """Process one road scene and save its result."""
    
    image_path = DATASET_DIR / image_name

    print(f"\nProcessing: {image_name}")

    image, detections = detect_objects(image_path)

    output_image = draw_detections(
        image,
        detections
    )

    output_name = (
        Path(image_name).stem +
        "_detected.jpg"
    )

    output_path = RESULTS_DIR / output_name

    cv2.imwrite(
        str(output_path),
        output_image
    )

    print(
        f"Detected {len(detections)} "
        f"road-related objects."
    )

    print(f"Saved result: {output_path}")

    for detection in detections:
        print(
            f'  {detection["object"]} '
            f'| confidence = '
            f'{detection["confidence"]} '
            f'| box = '
            f'{detection["bounding_box"]}'
        )


def main():
    """Run object detection on both road scenes."""
    
    process_image("road_scene.jpg")
    process_image("road_scene_2.jpg")

    print("\nGeneral object detection completed.")


if __name__ == "__main__":
    main()
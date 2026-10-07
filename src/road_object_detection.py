import json
from pathlib import Path

import cv2
from ultralytics import YOLO
from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

GENERAL_MODEL_NAME = "yolo11n.pt"
TRAFFIC_SIGN_REPO = "cvtechniques/TrafficSignDetection"

GENERAL_CONFIDENCE = 0.35
TRAFFIC_SIGN_CONFIDENCE = 0.20

# Larger inference size helps with small objects.
TRAFFIC_SIGN_IMAGE_SIZE = 960

# Tiling parameters for detecting small traffic signs.
TILE_SIZE = 640
TILE_OVERLAP = 0.25


# ============================================================
# GENERAL ROAD OBJECT CLASSES
# ============================================================

ROAD_OBJECT_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


# ============================================================
# LOAD GENERAL YOLO MODEL
# ============================================================

def load_general_model():
    print("Loading general YOLO model...")

    model = YOLO(GENERAL_MODEL_NAME)

    print("General YOLO model loaded successfully.")

    return model


# ============================================================
# FIND AND LOAD TRAFFIC SIGN MODEL
# ============================================================

def load_traffic_sign_model():
    print("\nFinding traffic-sign model...")

    repo_id = TRAFFIC_SIGN_REPO

    api = HfApi()

    files = api.list_repo_files(repo_id=repo_id)

    model_files = [
        file
        for file in files
        if file.lower().endswith(".pt")
    ]

    if not model_files:
        raise RuntimeError(
            "No .pt traffic-sign model was found in the Hugging Face repository."
        )

    # Prefer best.pt if available.
    best_models = [
        file
        for file in model_files
        if Path(file).name.lower() == "best.pt"
    ]

    if best_models:
        model_filename = best_models[0]
    else:
        model_filename = model_files[0]

    print(f"Traffic-sign model found: {model_filename}")

    print("Loading traffic-sign model...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename
    )

    model = YOLO(model_path)

    print("Traffic-sign model loaded successfully.")

    return model


# ============================================================
# GENERAL OBJECT DETECTION
# ============================================================

def detect_general_objects(model, image):
    print("\nRunning general object detection...")

    results = model.predict(
        source=image,
        conf=GENERAL_CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    predictions = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        if class_name not in ROAD_OBJECT_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        predictions.append({
            "object": class_name,
            "confidence": round(confidence, 3),
            "bounding_box": [
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ],
            "source": "general_yolo"
        })

    print(
        f"General objects detected: {len(predictions)}"
    )

    return predictions


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# NON-MAXIMUM SUPPRESSION FOR TRAFFIC SIGN TILES
# ============================================================

def remove_duplicate_signs(
    predictions,
    iou_threshold=0.45
):

    if not predictions:
        return []

    # Process each sign class separately.
    final_predictions = []

    classes = set(
        prediction["object"]
        for prediction in predictions
    )

    for class_name in classes:

        class_predictions = [
            prediction
            for prediction in predictions
            if prediction["object"] == class_name
        ]

        # Highest confidence first.
        class_predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        kept = []

        while class_predictions:

            current = class_predictions.pop(0)

            kept.append(current)

            remaining = []

            for candidate in class_predictions:

                iou = calculate_iou(
                    current["bounding_box"],
                    candidate["bounding_box"]
                )

                if iou < iou_threshold:
                    remaining.append(candidate)

            class_predictions = remaining

        final_predictions.extend(kept)

    final_predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    return final_predictions


# ============================================================
# TRAFFIC SIGN DETECTION USING IMAGE TILES
# ============================================================

def detect_traffic_signs(model, image):

    print("\nRunning high-resolution traffic-sign detection...")

    image_height, image_width = image.shape[:2]

    predictions = []

    stride = int(
        TILE_SIZE * (1 - TILE_OVERLAP)
    )

    tile_number = 0

    # --------------------------------------------------------
    # Generate overlapping image tiles.
    # --------------------------------------------------------

    y_positions = list(
        range(
            0,
            max(1, image_height - TILE_SIZE + 1),
            stride
        )
    )

    x_positions = list(
        range(
            0,
            max(1, image_width - TILE_SIZE + 1),
            stride
        )
    )

    # Make sure the final part of the image is included.
    if image_height > TILE_SIZE:
        final_y = image_height - TILE_SIZE

        if final_y not in y_positions:
            y_positions.append(final_y)

    if image_width > TILE_SIZE:
        final_x = image_width - TILE_SIZE

        if final_x not in x_positions:
            x_positions.append(final_x)

    # For smaller images, use the entire image.
    if image_height <= TILE_SIZE:
        y_positions = [0]

    if image_width <= TILE_SIZE:
        x_positions = [0]

    for y in y_positions:

        for x in x_positions:

            tile_number += 1

            y2 = min(
                y + TILE_SIZE,
                image_height
            )

            x2 = min(
                x + TILE_SIZE,
                image_width
            )

            tile = image[y:y2, x:x2]

            # ------------------------------------------------
            # Upscale tile.
            # ------------------------------------------------

            enlarged_tile = cv2.resize(
                tile,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )

            results = model.predict(
                source=enlarged_tile,
                conf=TRAFFIC_SIGN_CONFIDENCE,
                imgsz=TRAFFIC_SIGN_IMAGE_SIZE,
                verbose=False
            )

            result = results[0]

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                # Coordinates are from the enlarged tile.
                tx1, ty1, tx2, ty2 = (
                    box.xyxy[0].tolist()
                )

                # Convert coordinates back to original tile scale.
                scale = 1.5

                tx1 /= scale
                ty1 /= scale
                tx2 /= scale
                ty2 /= scale

                # Convert tile coordinates to full-image coordinates.
                full_x1 = x + tx1
                full_y1 = y + ty1
                full_x2 = x + tx2
                full_y2 = y + ty2

                # Clamp coordinates to image dimensions.
                full_x1 = max(
                    0,
                    min(full_x1, image_width)
                )

                full_y1 = max(
                    0,
                    min(full_y1, image_height)
                )

                full_x2 = max(
                    0,
                    min(full_x2, image_width)
                )

                full_y2 = max(
                    0,
                    min(full_y2, image_height)
                )

                predictions.append({
                    "object": class_name,
                    "confidence": round(
                        confidence,
                        3
                    ),
                    "bounding_box": [
                        round(full_x1),
                        round(full_y1),
                        round(full_x2),
                        round(full_y2)
                    ],
                    "source": "traffic_sign_yolo"
                })

    # Remove duplicate detections created by overlapping tiles.
    predictions = remove_duplicate_signs(
        predictions,
        iou_threshold=0.45
    )

    print(
        f"Traffic signs detected: {len(predictions)}"
    )

    return predictions


# ============================================================
# DRAW PREDICTIONS
# ============================================================

def draw_predictions(
    image,
    predictions
):

    output = image.copy()

    for prediction in predictions:

        x1, y1, x2, y2 = (
            prediction["bounding_box"]
        )

        object_name = prediction["object"]

        confidence = prediction["confidence"]

        source = prediction["source"]

        # Different colors for general objects
        # and traffic signs.
        if source == "traffic_sign_yolo":

            box_color = (0, 0, 255)

        else:

            box_color = (0, 255, 0)

        label = (
            f"{object_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        text_y = max(
            y1 - 8,
            20
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    return output


# ============================================================
# SAVE STRUCTURED JSON
# ============================================================

def save_predictions(
    predictions,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            predictions,
            file,
            indent=4
        )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_name,
    general_model,
    traffic_sign_model
):

    print("\n")
    print("=" * 60)
    print(
        f"PROCESSING IMAGE: {image_name}"
    )
    print("=" * 60)

    image_path = DATASET_DIR / image_name

    if not image_path.exists():

        print(
            f"WARNING: Image not found: {image_path}"
        )

        return None

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"ERROR: Could not load image: {image_path}"
        )

        return None

    print("Image loaded successfully.")

    # --------------------------------------------------------
    # General object detection
    # --------------------------------------------------------

    general_predictions = detect_general_objects(
        general_model,
        image
    )

    # --------------------------------------------------------
    # Traffic sign detection
    # --------------------------------------------------------

    sign_predictions = detect_traffic_signs(
        traffic_sign_model,
        image
    )

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    all_predictions = (
        general_predictions +
        sign_predictions
    )

    # --------------------------------------------------------
    # Create annotated image
    # --------------------------------------------------------

    annotated_image = draw_predictions(
        image,
        all_predictions
    )

    image_stem = Path(
        image_name
    ).stem

    output_image_path = (
        RESULTS_DIR /
        f"{image_stem}_final_detected.jpg"
    )

    output_json_path = (
        RESULTS_DIR /
        f"{image_stem}_predictions.json"
    )

    cv2.imwrite(
        str(output_image_path),
        annotated_image
    )

    save_predictions(
        all_predictions,
        output_json_path
    )

    # --------------------------------------------------------
    # Print structured predictions
    # --------------------------------------------------------

    print("\nStructured predictions:")

    if not all_predictions:

        print("No objects or signs detected.")

    else:

        for index, prediction in enumerate(
            all_predictions,
            start=1
        ):

            print(
                f"\n{index}. Object: "
                f"{prediction['object']}"
            )

            print(
                f"   Confidence: "
                f"{prediction['confidence']}"
            )

            print(
                f"   Bounding Box: "
                f"{prediction['bounding_box']}"
            )

            print(
                f"   Source: "
                f"{prediction['source']}"
            )

    print(
        f"\nTotal objects/signs detected: "
        f"{len(all_predictions)}"
    )

    print(
        "\nAnnotated image saved to:"
    )

    print(output_image_path)

    print(
        "\nStructured JSON saved to:"
    )

    print(output_json_path)

    return {
        "image": image_name,
        "detections": len(all_predictions),
        "output_image": str(
            output_image_path
        ),
        "output_json": str(
            output_json_path
        )
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print(
        "CNN-BASED ROAD MONITORING SYSTEM"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    general_model = load_general_model()

    print("\nFinding traffic-sign model...")

    traffic_sign_model = (
        load_traffic_sign_model()
    )

    # --------------------------------------------------------
    # Images to process
    # --------------------------------------------------------

    image_names = [
        "road_scene.jpg",
        "road_scene_2.jpg",
        "road_scene_3.jpg"
    ]

    summaries = []

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    for image_name in image_names:

        result = process_image(
            image_name,
            general_model,
            traffic_sign_model
        )

        if result is not None:

            summaries.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(
        "ROAD MONITORING DETECTION COMPLETED"
    )
    print("=" * 60)

    print(
        f"\nImages processed: "
        f"{len(summaries)}"
    )

    for summary in summaries:

        print(
            f"  {summary['image']}: "
            f"{summary['detections']} detections"
        )

    print(
        f"\nResults folder:"
    )

    print(RESULTS_DIR)

    print(
        "\nProject execution completed successfully."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
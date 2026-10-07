# CNN-Based Road Monitoring System

An academic prototype for automated detection and structured analysis of road users and traffic infrastructure from road-scene imagery using Convolutional Neural Network (CNN) and YOLO architectures.

---

## 1. Project Overview

The **CNN-Based Road Monitoring System** is an image-based computer vision prototype designed to monitor roadway environments by detecting key road participants (vehicles and pedestrians) alongside critical traffic infrastructure (traffic signs).

The system integrates two complementary object detection models:
1. **General Road-Object Detector:** Powered by the state-of-the-art **YOLO11n** architecture to detect common traffic entities such as cars, pedestrians, buses, and trucks.
2. **Specialized Traffic-Sign Detector:** A domain-specific YOLO model fetched from the **Hugging Face Hub** (`cvtechniques/TrafficSignDetection`) dedicated to identifying regulatory and warning road signs (e.g., STOP signs).

For every input image, the system generates:
- **Annotated Images:** Bounding boxes, class names, and confidence scores rendered onto the input frame.
- **Structured JSON Predictions:** Machine-readable records detailing detected objects, confidence values, pixel coordinates, and model source.
- **Interactive Notebook:** A Jupyter notebook demonstrating step-by-step pipeline execution, qualitative analysis, and visualizations.

---

## 2. Problem Statement

Automated road monitoring is an essential component of Intelligent Transportation Systems (ITS) and Advanced Driver Assistance Systems (ADAS). Accurately detecting road participants and traffic signs presents multiple computer vision challenges:
- **Scale Variation:** Vehicles and pedestrians occupy large image areas, whereas traffic signs are frequently small, distant, and easily missed by standard general-purpose detectors.
- **Dense Urban Clutter:** Occlusions and overlapping objects in busy urban scenes complicate localization.
- **Safety and Infrastructure Compliance:** Detecting traffic signs requires specialized feature representations that general multi-class datasets (such as COCO) do not specialize in.

This project addresses these challenges by combining a lightweight general detector with a specialized sign detector to create a unified, structured road monitoring pipeline.

---

## 3. Objective

The primary objectives of this project are:
- Develop a modular CNN/YOLO-based road monitoring prototype for static road scenes.
- Detect and localize key general road entities: `person`, `car`, `truck`, `bus`, `motorcycle`, and `bicycle`.
- Detect and classify road traffic signs (e.g., STOP signs) using high-resolution inspection and tiling strategies.
- Aggregate detections from multiple models into a clean, deduplicated structured output format.
- Export visual detection outputs and structured JSON files for downstream analysis.
- Provide comprehensive academic documentation and interactive Jupyter notebook demonstrations suitable for coursework evaluation and viva discussion.

---

## 4. Methodology

The pipeline follows a multi-stage detection and fusion methodology:

1. **Model Selection & Initialization:**
   - **General Detection:** Ultralytics **YOLO11n** (`yolo11n.pt`) is loaded locally. Inference runs at 640×640 resolution with a confidence threshold of 0.35. Predictions are filtered strictly to the road-relevant classes: `person`, `car`, `truck`, `bus`, `motorcycle`, and `bicycle`.
   - **Traffic-Sign Detection:** The specialized model `best.pt` is retrieved dynamically from Hugging Face (`cvtechniques/TrafficSignDetection`). Inference runs at a higher resolution (960×960) with a confidence threshold of 0.20 to capture small signs. Optional image tiling (640×640 tiles with 25% overlap) ensures localized sign features are preserved.

2. **Prediction Fusion & Deduplication:**
   - Predictions from both models are collected into structured records.
   - An Intersection-over-Union (IoU) calculation is applied to prevent redundant duplicate bounding boxes between detectors.

3. **Visualization & Persistence:**
   - Detections are drawn onto the original image using OpenCV with distinct bounding box colors and labels.
   - Structured predictions are serialized and saved to formatted JSON files in the `results/` directory.

---

## 5. System Workflow

```
                        +----------------------------+
                        |      Input Road Image      |
                        |       (dataset/*.jpg)      |
                        +--------------+-------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |      General YOLO       |             |   Traffic-Sign YOLO     |
      |        (YOLO11n)        |             |      (Hugging Face)     |
      |   conf: 0.35, imgsz: 640|             |   conf: 0.20, imgsz: 960|
      +------------+------------+             +------------+------------+
                   |                                       |
                   | Road-Class Filtering                  | High-Res / Tiled
                   | (car, person, bus, etc.)              | Sign Localization
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                        +----------------------------+
                        |     Prediction Fusion      |
                        |      & IoU Filtering       |
                        +--------------+-------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |  Annotated Visual Image |             | Structured JSON Record  |
      |   (results/*_detected)  |             |  (results/*_predictions)|
      +-------------------------+             +-------------------------+
```

---

## 6. Technologies Used

- **Python 3.10+ / 3.13:** Primary programming language.
- **Ultralytics YOLO:** Deep learning framework used for loading and executing YOLO11 and YOLO-based sign models.
- **Hugging Face Hub (`huggingface_hub`):** Dynamic retrieval of the specialized traffic-sign model weights.
- **OpenCV (`opencv-python`):** Image loading, color conversions, coordinate scaling, and visual bounding box rendering.
- **NumPy:** Numerical operations and array manipulation.
- **Matplotlib:** Visualization and figure generation within the Jupyter notebook.
- **Jupyter Notebook:** Interactive experimentation, visual walkthrough, and demonstration environment.

---

## 7. Project Structure

```
road-monitoring-cnn/
├── dataset/
│   ├── road_scene.jpg                  # Test image 1 (busy urban intersection)
│   ├── road_scene_2.jpg                # Test image 2 (urban street scene)
│   └── road_scene_3.jpg                # Test image 3 (wide road with traffic and pedestrians)
├── notebooks/
│   └── Road_Object_Detection.ipynb     # Interactive Jupyter notebook demonstration
├── results/
│   ├── road_scene_final_detected.jpg   # Annotated detection output for scene 1
│   ├── road_scene_2_final_detected.jpg # Annotated detection output for scene 2
│   ├── road_scene_3_final_detected.jpg # Annotated detection output for scene 3
│   ├── road_scene_predictions.json     # Structured JSON predictions for scene 1
│   ├── road_scene_2_predictions.json   # Structured JSON predictions for scene 2
│   └── road_scene_3_predictions.json   # Structured JSON predictions for scene 3
├── screenshots/
│   └── structured_predictions.png      # Terminal output screenshot of structured predictions
├── src/
│   └── road_object_detection.py        # Main Python detection and inference script
├── .gitignore                          # Excludes venv, weights (*.pt), and cache files
├── README.md                           # Complete project documentation
├── requirements.txt                    # Project dependency specifications
└── yolo11n.pt                          # Local general YOLO model weights (ignored by git)
```

---

## 8. Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd road-monitoring-cnn
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Running the Project

### Option A: Run the Standalone Python Pipeline
Execute the main detection script from the repository root:
```bash
python src/road_object_detection.py
```
The script will:
1. Initialize the general YOLO11n model.
2. Connect to the Hugging Face Hub and load the specialized traffic-sign model.
3. Process all three road scenes in `dataset/`.
4. Output structured predictions directly to the terminal.
5. Save the final annotated images and structured JSON files into `results/`.

### Option B: Interactive Jupyter Notebook
Launch the Jupyter Notebook environment to inspect the pipeline interactively:
```bash
jupyter notebook notebooks/Road_Object_Detection.ipynb
```
Follow the sequential cells to view model loading, detection procedures, side-by-side visual comparisons, and structured dictionary outputs.

---

## 10. Results

The prototype was tested on three diverse road-scene images. The empirical detection outcomes are summarized below:

| Test Image | Dimensions | General Objects Detected | Traffic Signs Detected | Total Detections | Key Observed Classes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `road_scene.jpg` | 547 × 365 | 18 | 1 (`stop`) | **19** | Pedestrians (12), Cars (2), Buses (2), Truck (1), STOP Sign (1) |
| `road_scene_2.jpg` | 646 × 475 | 10 | 0 | **10** | Pedestrians (7), Buses (2), Car (1) |
| `road_scene_3.jpg` | 1324 × 739 | 11 | 0 | **11** | Cars (7), Pedestrians (4) |

### Key Observations:
- **`road_scene.jpg`:** Successfully localized a prominent regulatory **STOP sign** (confidence: 0.628, bounding box: `[416, 214, 439, 243]`) alongside 18 pedestrians and vehicles.
- **`road_scene_2.jpg` & `road_scene_3.jpg`:** Produced robust multi-object detections for vehicles and pedestrians. No false-positive traffic signs were introduced where clear signs were absent.
- **Evaluation Note:** No overall accuracy percentage or mean Average Precision (mAP) is claimed here because this prototype was qualitatively demonstrated on representative sample scenes rather than evaluated against a formal, annotated ground-truth benchmark test set.

---

## 11. Output Format

Detections are exported to `results/<image_name>_predictions.json` as an array of structured JSON objects:

```json
[
  {
    "object": "person",
    "confidence": 0.859,
    "bounding_box": [120, 180, 179, 294],
    "source": "general_yolo"
  },
  {
    "object": "car",
    "confidence": 0.51,
    "bounding_box": [80, 179, 164, 264],
    "source": "general_yolo"
  },
  {
    "object": "stop",
    "confidence": 0.628,
    "bounding_box": [416, 214, 439, 243],
    "source": "traffic_sign_yolo"
  }
]
```

### Field Definitions:
- `object` *(str)*: Identified class label.
- `confidence` *(float)*: Softmax detection confidence score (0.0 to 1.0).
- `bounding_box` *(list[int])*: Pixel coordinates formatted as `[x1, y1, x2, y2]` (top-left and bottom-right corners).
- `source` *(str)*: Model responsible for the detection (`general_yolo` or `traffic_sign_yolo`).

---

## 12. Limitations

- **Small and Distant Objects:** Highly distant or low-resolution traffic signs may fall below the detection threshold.
- **Weather and Lighting Conditions:** The models were evaluated on daytime, clear-weather imagery. Performance may degrade under adverse weather (heavy rain, snow, dense fog) or nighttime lighting without domain-specific data augmentation.
- **Single-Frame Static Processing:** The current prototype operates on static images sequentially rather than leveraging temporal tracking across video frames.
- **Prototype Status:** This system is an academic prototype intended for demonstration and coursework evaluation; it is not designed or validated for safety-critical deployment in autonomous vehicles or live traffic control systems.

---

## 13. External Resources and Acknowledgements

This project builds upon the following open-source tools and models:
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics):** Deep learning object detection framework and pre-trained YOLO11 models.
- **[Hugging Face Hub](https://huggingface.co/):** Model hosting and distribution platform.
- **Traffic Sign Model:** `cvtechniques/TrafficSignDetection` on Hugging Face Hub.
- **[OpenCV](https://opencv.org/):** Computer vision library for image processing.
- **[NumPy](https://numpy.org/) & [Matplotlib](https://matplotlib.org/):** Numerical operations and visual rendering.
- **[Jupyter](https://jupyter.org/):** Interactive computing platform.

---

## 14. Academic Purpose

This project was developed as an individual academic coursework submission in computer vision and deep learning. It demonstrates:
- Practical application of Convolutional Neural Networks and modern YOLO architectures.
- Multi-model fusion techniques for combining general and specialized object detectors.
- Structured data extraction and computer vision pipeline design.
- Rigorous experimentation, visualization, and documentation practices.

---

## 15. Author

- **Author:** Rejeena Agastian
- **Project:** CNN-Based Road Monitoring System
- **Context:** Academic Coursework / Individual Assignment Submission

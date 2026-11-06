# Eye Cat v3.2

> Real-time kitten detection and segmentation on an RTSP camera stream, featuring a "cat escape" alert.

## About the Project

This project was developed for the Object-Oriented Programming Laboratory course, taught by Professor Derzu Omaia, as part of the Computer Engineering undergraduate program at UFPB.

The assignment was to build any computer vision application in C++. I wanted to tackle something more complex, so I dedicated extra effort to this specific project, drawing on prior experience from another project involving OpenCV.

## Motivation

Around the time I received the assignment, my cat Marie had just given birth to a litter, and it was not her first time having kittens. The previous time, one of them managed to get out of the box, suffered an accident, and unfortunately passed away.

To help prevent this from happening again, I came up with the idea of building an application using a Wi-Fi camera to detect when kittens escape from the box they are kept in.

## Hardware

I purchased an "Orbtronic ORB7160" Wi-Fi camera on Amazon—a simple, budget-friendly camera (around R$ 60) featuring Wi-Fi connectivity and USB-C power.

When it arrived, I noticed it came with proprietary software called "YooSee," which I found quite clunky and restrictive in terms of customization. However, because it is a Wi-Fi camera, I was able to configure it for direct access using only its local IP address and a custom password I set.

I mounted it onto a wire mesh grid above the kitten box, providing a full view of all the kittens as well as the area outside the box. I connected a 10,000 mAh power bank capable of powering the setup for nearly 24 hours.

With that, I had all the hardware needed for the program to function and could begin working on the software—and that is when the first obstacle arose.

## Challenges and Approach Selection

As seen in some of the dataset images, the kittens huddled closely together on top of one another—there were 6 kittens in total. My initial idea was to trigger an alarm whenever the program detected an average of fewer than 6 kittens inside the box over a given period X. However, if a kitten were sleeping underneath the others and became completely hidden from the camera, the system would misinterpret this as a false escape.

Another issue I anticipated was that when the kittens slept tangled up together, conventional OpenCV models would struggle to identify them. Standard bounding box models, which are extremely common, do not work well in this scenario because it is impossible to draw a box containing only a single kitten. Often, a kitten was not fully visible to the camera—only a tail or head might poke out—so I knew a bounding box model would not suffice.

I tested pre-trained bounding box models, but none worked reliably for this scenario: many detected over 10 cats or none at all, and pre-trained segmentation models also failed. After spending substantial time trying to adapt pre-trained solutions, I realized it would be better to train a custom model tailored specifically to my use case.

I encountered a few hurdles while creating and setting up the dataset, but by adhering to best practices for dataset curation and training on the Ultralytics platform, I achieved a satisfactory result.

## Dataset Construction

Training a model on this dataset required specific strategies. Early on, I made mistakes that introduced bias into the model: the initial photos were taken all on a single day, with the same blanket inside the box, identical lighting, the same objects, and similar blanket folds/wrinkles. As a result, when the model encountered a different fold pattern, it would misclassify a fold of fabric that resembled a tail or paw as a kitten.

Another mistake was including only images containing kittens and consistently having the exact same number of kittens in each image. This caused the model to hallucinate, assuming kittens were always present or that there should always be 6 kittens in every frame.

Ultimately, I curated a robust dataset with over 450 images, resolving every previous error.

## Model Training

The training run on Ultralytics yielded the following metrics:

| Metric | Value |
|---|---|
| mAP50(M) | 99.3% |
| mAP50-95(M) | 81.8% |
| Precision(M) | 98.1% |
| Recall(M) | 97.9% |

A brief explanation of each metric (the "(M)" denotes that metrics are calculated based on the segmentation mask, not just the bounding box):

- **Precision**: Out of everything the model labeled as a "kitten," the percentage that actually was a kitten. High precision indicates very few false positives (the model does not hallucinate kittens where none exist).
- **Recall**: Out of all the kittens actually present in the image, the percentage the model successfully identified. High recall indicates very few false negatives (the model does not miss any kittens).
- **mAP50**: Mean Average Precision where a prediction is counted as correct if the predicted segmentation mask has at least 50% overlap (Intersection over Union - IoU) with the ground truth mask. This is the standard benchmark metric for object localization.
- **mAP50-95**: The same concept as mAP50, but evaluated across multiple IoU overlap thresholds (from 50% to 95%, in steps) and averaged. This is a much stricter metric, requiring the predicted mask to match the precise geometric outline of the kitten rather than just roughly covering it.

## Model Conversion to NCNN

The trained model was exported from Ultralytics as `exp.pt`, which is a PyTorch weights file. The issue is that PyTorch is not intended for this runtime scenario: it is resource-heavy, depends on Python libraries and CUDA (or ROCm) for GPU acceleration, and does not run smoothly in pure C++ inside a real-time video processing loop.

Since the final application needed to run in C++ with low latency and GPU acceleration on an AMD Radeon RX 6650 XT (which lacks official CUDA support and has limited ROCm support on consumer GPUs), the solution was to use **NCNN**. Developed by Tencent, NCNN is a lightweight C++ inference framework that runs via **Vulkan**—allowing it to run on any GPU with a Vulkan driver, including AMD graphics cards.

However, NCNN cannot read `.pt` files directly. Converting the model required a two-step process:

1. Export `exp.pt` to an intermediate format (TorchScript/ONNX) using Ultralytics' built-in export tool;
2. Convert that intermediate format to NCNN's native format using **PNNX** (PyTorch Neural Network eXchange), a bridging tool created by the author of NCNN.

The result of this conversion is the `exp_ncnn_model/` directory, containing the following files:

- `model.ncnn.param` — The network architecture (layer definitions and connections)
- `model.ncnn.bin` — The trained model weights
- `metadata.yaml` and `model_ncnn.py` — Automatically generated during export, containing model metadata

The application (`mainT.cpp`) loads these first two files (`.param` and `.bin`) directly to perform inference.

## Features

### Real-Time Detection and Segmentation

The program's core: for every frame received from the camera via RTSP, the trained model detects each kitten and delineates its exact boundary (segmentation) rather than drawing a simple box. A temporal tracker confirms detection across consecutive frames before counting or rendering the kitten on screen, filtering out single-frame hallucinations.

### Cat Escape Alert

The primary safeguard. Whenever the system detects a kitten outside the designated box area and the target kitten count "inside" has not been satisfied, it:

- Displays a flashing on-screen warning ("FUGA DE GATO!");
- Sounds a recurring beep while the alert remains active;
- Triggers `toque.sh`, which uses a KDE Connect tool (GSConnect) to ring my phone.

One challenge was that this notification only reached my phone if both devices were on the same local network. I resolved this easily by connecting my phone and computer via **ZeroTier**, allowing notifications to be delivered even when I am away from home.

### Capture Photo

Pressing the **'f'** key while the application window is focused saves a raw frame from the camera—the original image without any detection overlays drawn on top. The goal is to capture clean records of the kittens (for expanding the dataset, tracking their growth, etc.) directly from the monitoring software without needing to hold a smartphone.

### Reporting

When exiting the application (via the 'q' key or ESC), it automatically generates a report summarizing the kitten count history recorded during runtime:

- A PNG chart drawn from scratch using OpenCV functions alone (without external plotting dependencies), displaying the kitten count over time as a step plot;
- A CSV file containing the raw data (timestamp in seconds vs. kitten count), allowing for analysis in external tools like Excel or Python.

The purpose is to review behavioral patterns throughout the day and night, such as identifying when kittens tend to leave the box most frequently.

## Demo

https://github.com/user-attachments/assets/a90e9031-0930-4c30-8767-e6e01c1aaf38

## Repository Structure

```
.
├── exp_ncnn_model/
│   ├── metadata.yaml
│   ├── model.ncnn.bin       # trained model weights
│   ├── model.ncnn.param     # model architecture
│   └── model_ncnn.py
├── exp.pt                  # original model exported by Ultralytics (PyTorch)
├── mainT.cpp                # core application source code
├── toque.sh                 # fallback script to trigger mobile notification
├── gato_seg3_toque          # compiled binary
└── prints/                  # screenshots / raw captures saved by the program
```

## Dependencies

- **OpenCV** (version 5 — uses the `geometry` module for `contourArea`/`approxPolyDP`)
- **NCNN**, built with Vulkan support
- **Vulkan SDK** + GPU Vulkan driver (e.g., `vulkan-radeon` or `amdvlk` for AMD on Linux)
- **FFmpeg** (required by OpenCV to open RTSP streams)
- **CMake** (3.16 or higher) and a C++17-compliant compiler

> ⚠️ Adjust with the exact versions used during development if you want greater precision.

## Building from Source

The project uses **CMake**. The `CMakeLists.txt` file is located at the root of the repository alongside `mainT.cpp`.

### 1. Install System Dependencies

On Arch Linux, for example:

```bash
sudo pacman -S cmake opencv ffmpeg vulkan-headers vulkan-icd-loader vulkan-radeon vulkan-tools
```

Replace `vulkan-radeon` with the appropriate Vulkan driver package for your GPU (AMD, Intel, NVIDIA, etc.), and `opencv` if your distribution lacks OpenCV 5 in its official package repositories (in which case, build OpenCV from source).

Verify that Vulkan is working properly before continuing:

```bash
vulkaninfo
```

### 2. Build NCNN with Vulkan Support

Most Linux distributions do not package NCNN, so it is typically necessary to build it directly from the official repository:

```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
mkdir build && cd build
cmake -DNCNN_VULKAN=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
```

This generates an `install/` directory inside `build/` containing the NCNN headers and precompiled libraries ready for integration—this is the path to supply to `CMakeLists.txt` in the next step.

### 3. Build the Application

From the root of this repository (where `mainT.cpp` and `CMakeLists.txt` are located):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/build/install/lib/cmake/ncnn
cmake --build build -j$(nproc)
```

Replace `/path/to/ncnn/build/install` with the actual path where you compiled and installed NCNN in the previous step. If CMake locates NCNN automatically (e.g., if installed system-wide in `/usr/local`), the `-Dncnn_DIR=...` flag can be omitted.

Upon completion, the compiled binary will be located at `build/eye_cat_v3`.

### 4. Configuration and Execution

Before running, update the camera URL in `mainT.cpp` and recompile:

```cpp
static const char* RTSP_URL = "rtsp://username:password@LOCAL_IP:554/onvif1";
```

Also ensure that the `exp_ncnn_model/` directory (containing `model.ncnn.param` and `model.ncnn.bin`) is in the working directory from which the binary is executed.

```bash
cd build
./eye_cat_v3
```

## Runtime Controls

| Key | Action |
|---|---|
| `f` | Saves a raw photo (no annotations) from the camera |
| `q` or `ESC` | Exits the program and generates the report (PNG + CSV) |

During startup, the program prompts you to draw and select the monitored box area using the mouse (ENTER or Space confirms the selection, while ESC disables escape monitoring).

## Credits

- Base model: YOLO-seg architecture, by [Ultralytics](https://www.ultralytics.com/);
- Conversion to NCNN: [NCNN](https://github.com/Tencent/ncnn) and [PNNX](https://github.com/pnnx/pnnx) (Tencent);
- Dataset and training: [Custom dataset collected and trained by the project author](https://platform.ultralytics.com/gabriel-ch/datasets/eye-on-cat-v1).
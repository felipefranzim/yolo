# 📝 YOLO Model Training and Testing Guide

This repository provides a comprehensive guide for setting up your environment, training, and testing YOLO models. It is designed to help users leverage GPU acceleration for efficient deep learning workflows.

## 📋 Requirements

> If you do not intend to use the GPU, go to item 4.

1. Always install the latest driver for your GPU.
2. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local). CUDA 12.1 is the recommended one.
3. Install [`cuDNN`](https://developer.nvidia.com/cudnn).
4. Install the required packages in your Python virtual environment:

    ```bash
    python -m venv yolo-env # Create the virtual environment
    yolo-env\Scripts\activate # Activate the virtual environment
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics

    # If training freezes after the first epoch and does not complete, it might be due to an incompatibility with the `polars` package and your CPU. If so, uninstall and install the legacy CPU version of `polars`:
    pip uninstall polars
    pip install polars-lts-cpu

    # Or if still does not work and the terminal shows some message like "The following required CPU features were not detected: avx2, fma, bmi1, bmi2, lzcnt, movbe Continuing to use this version of Polars on this processor will likely result in a crash. Install polars[rtcompat] instead of polars to run Polars with better compatibility." try:

    pip uninstall polars
    pip install "polars[rtcompat]"
    ```

    > **Note:** The URL in `--index-url https://download.pytorch.org/whl/cu121` may vary depending on your installed CUDA version. Refer to the [PyTorch documentation](https://pytorch.org/get-started/locally/) to find the appropriate installation for your CUDA version.

## 🏋️ Training the Model

To start training the model, use the following command:

    ```bash
    yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 workers=0
    ```

## 🧪 Testing the Model

After training, the results will be available in the `runs/detect/train/` directory.

To test the trained model, run:

    ```bash
    yolo detect predict model=runs/detect/train/weights/best.pt source=horse_test.jpg
    ```

Or using the ONNX model:

    ```bash
    yolo detect predict model=runs/detect/train/weights/best.onnx source=horse_test.jpg
    ```

## Convert to ONNX

If you want to converter the result model to the ONNX format

```bash
    yolo export model=runs/detect/train/weights/best.pt format=onnx
```
# Model Description

This repository provides a model consisting of three main components:

1. **Text Detection**
2. **Image-to-Text Conversion**
3. **Character Tagging (Named Entity Recognition - NER)**

## Configuration and Usage Instructions

### Configuration

1. **Default Values**  
   Default values are provided in the `config` file. You can configure the model to use YOLO for speech detection by setting the `model` parameter to `Yolo`. Specify the path to the YOLO model in the `modelpath` parameter. If you prefer not to use YOLO, set the `model` to `simple`.

2. **Model Path**  
   Ensure that the `modelpath` parameter in the configuration file is correctly set to the path of your YOLO model if using YOLO for speech detection.

3. **Performance Considerations**  
   - **GPU Acceleration:** The model performs faster with a GPU. To use GPU acceleration, uninstall your current PyTorch version and install the CUDA-enabled version from the [PyTorch website](https://pytorch.org/get-started/locally/).
   - **CPU Option:** If a GPU is not available, you can continue using the CPU version of PyTorch.

4. **Number of images in the folder**
   Ensure that the number of images in the folder are not too many

### Running the Model

1. **Update Configuration**  
   Modify the `config` file with your specific parameters and model paths.

2. **Execution**  
   Run the main script using the command:
   ```bash
   python main.py

# Super-Resolution of Satellite Images using Deep Learning

This repository contains the implementation and evaluation of multiple *Image Super-Resolution (SR)* models applied to satellite images. The primary focus is on a fine-tuned *Real-ESRGAN, with a comparative analysis against other prominent models: **SRCNN, LAPSRN, RCAN, and SRGAN*.

The goal is to enhance the resolution of low-quality satellite imagery, which has significant applications in fields like environmental monitoring, urban planning, agriculture, and disaster management. 🛰

---

## 📂 Repository Structure

├── esrgan_inference.py  # Script for running inference on test images
├── models/              # Folder for model weights and configs
│   └── Model weights download.txt
├── LR/                  # Folder for Low-Resolution input images
├── HR_reference/        # Ground Truth High-Resolution images
├── results/             # Generated High-Resolution results from inference
└── README.md            # Documentation

---

## 🧠 Models Overview

This project evaluates several deep learning models for super-resolution, each with a unique architecture and approach.

* *SRCNN (Super-Resolution Convolutional Neural Network):* One of the pioneering deep learning models for SR. It uses a simple three-layer convolutional network to learn the end-to-end mapping from low-resolution to high-resolution images.
* *LAPSRN (Laplacian Pyramid Super-Resolution Network):* A progressive upsampling model that reconstructs the high-resolution image in stages, predicting high-frequency residuals at each level of a Laplacian pyramid.
* *SRGAN (Super-Resolution Generative Adversarial Network):* Utilizes a Generative Adversarial Network (GAN) to generate more photorealistic images. It introduces a perceptual loss function that prioritizes human perception of image quality over traditional metrics like PSNR.
* *RCAN (Residual Channel Attention Network):* A very deep CNN model that achieves high performance by incorporating a residual-in-residual structure and a channel attention mechanism, allowing it to focus on more useful feature channels.
* *Real-ESRGAN (Fine-tuned):* The primary model of this project. It's an enhancement of ESRGAN (Enhanced SRGAN) trained on a diverse set of real-world images with complex degradations. Our version is fine-tuned specifically for satellite imagery to achieve state-of-the-art results.

---

## ⚙ Setup and Installation

Follow these steps to set up the project environment and run the models.

### Setup Flowchart
mermaid
graph TD
    A[Start] --> B{Clone Repository};
    B --> C{Install Dependencies};
    C --> D{Download Model Weights};
    D --> E[Ready to Run Inference ✅];

1. Prerequisites

   - Python 3.8+

   - PyTorch 1.7+

   - Git

2. Clone the Repository

Bash
git clone [https://github.com/NinadGawali/Satellite-Image-Super-Resolution-.git](https://github.com/NinadGawali/Satellite-Image-Super-Resolution-.git)
cd Satellite-Image-Super-Resolution-

3. Install Dependencies

Install the required Python packages. The primary dependency is realesrgan.
Bash
pip install numpy opencv-python torch torchvision torchaudio
pip install realesrgan


4. Download Pre-trained Models

The model weights are required to run inference. Download the necessary .pth files using the links provided in the models/Model weights download.txt file and place them in the /models directory.

The key model for this project is RealESRGAN_x4plus.pth.

🚀 How to Run Inference

You can easily generate super-resolved images using the provided inference script.

Inference Pipeline


graph TD
    A[Start: Place LR images in `/LR` folder] --> B[Run `esrgan_inference.py`];
    B -- Input LR Image & Model Path --> C{Real-ESRGAN Model};
    C -- Processes Image --> D{Generate Super-Resolved Image};
    D -- Save Image --> E[Output saved in `/results` folder];
    E --> F[End];

1. Prepare Your Data

    Place your low-resolution (LR) input images in the LR/ folder.

    The script will automatically create the output folder if it doesn't exist.

2. Run the Script

Execute esrgan_inference.py from your terminal. Use the following command as a template:
Bash
python esrgan_inference.py --input LR/ --output results/RealESRGAN_output --model_path models/RealESRGAN_x4plus.pth --scale 4

Key Arguments:

    --input: Path to the folder containing your low-resolution images.

    --output: Path to the folder where the results will be saved.

    --model_path: Path to the pre-trained model weights file (e.g., RealESRGAN_x4plus.pth).

    --scale: The upscaling factor (e.g., 4 for 4x super-resolution).

The generated images will be saved in the specified output directory.

📊 Results and Comparison

Here is a visual comparison of the results produced by different models on a sample satellite image. Our fine-tuned Real-ESRGAN demonstrates superior performance in reconstructing fine details and textures compared to other models.
```

```
For more visual examples from all tested models, please check the results/ directory.

🙏 Acknowledgements

This project builds upon the excellent work and open-source contributions of the research community.

    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data by Xintao Wang et al.

    BasicSR: An open-source image and video restoration toolbox that provides the foundation for many SR models.

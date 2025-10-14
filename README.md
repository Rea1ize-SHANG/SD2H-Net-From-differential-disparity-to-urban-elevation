# SD²H-Net-From-differential-disparity-to-urban-elevation

We introduce the Stereo Differential Disparity to Height Estimation Network (SD²H-Net), a pioneering end-to-end architecture that elegantly integrates classical differential parallax theory with deep learning paradigms.

## **How to use?**

### **Environment**
- Python 3.8.20
- Pytorch 2.0.1+cu117
- torchvision 0.15.2+cu117

### **Install**
1. Create a virtual environment and activate it:
    ```bash
    conda create -n SD2H python=3.8
    conda activate SD2H
    ```

2. Install dependencies:
    ```bash
    pip install --upgrade pip setuptools
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
    pip install tqdm 
    pip install scipy 
    pip install opencv-python 
    pip install scikit-image 
    pip install tensorboard 
    pip install matplotlib 
    pip install timm==0.6.13 
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
    pip install accelerate==1.0.1 
    pip install gradio_imageslider 
    pip install gradio==4.29.0 
    pip install hydra-core
    pip install opt-einsum
    pip install wandb 
    pip install scikit-learn
    pip install rasterio
    ```

## **Data Preparation**
Download the **[WHU-Stereo Datasets](https://github.com/Sheng029/WHU-Stereo)** and **[GF7Stereo Datasets]()**.

## **Appendix**
Supplementary data to this article can be found online at [Supplementary Data Link].


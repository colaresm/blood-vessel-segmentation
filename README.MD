# Blood Vessel Segmentation
This is a computer vision project focused on blood vessel segmentation in medical images using a U-Net architecture trained with TensorFlow/Keras in Python, and later converted to TensorFlow Lite for efficient inference in a C++ application.

The goal is to build a complete and optimized pipeline, from data preprocessing to embedded inference using TFLite.

📌 Objectives
- Train a U-Net neural network for blood vessel segmentation
- Export the trained model to .tflite format
- Integrate and run inference with TensorFlow Lite in C++
- Ensure performance and portability for embedded systems

⚙️ Technologies Used
- Python 
- TensorFlow / Keras
- U-Net (segmentation architecture)
-  TensorFlow Lite
- C++ (with CMake)
- OpenCV (for image handling in C++)

# Dataset
The dataset used in this project was the retinal blood vessel segmentation dataset available on [kaggle](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel). This set contains images of the fundus of the eye and their respective annotations, thus having 50 images in total.
<p align="center">
  <img src="images/sample.png" alt="image" width="300"/>
  <img src="images/sample_mask.png" alt="mask" width="300"/>
</p>

# Model conversion to TFLite
To convert the model trained by tensorflow into a .tflite model, simply run the ``create_tf_lite_model.py`` file with the following command:
```
python create_tf_lite_mode.py
```
The ``.tflite`` file will be saved in the models folder with the name ``segment_model.tflite``.

#  🚀 Get started
- Clone the repository:
```
git clone https://github.com/colaresm/blood-vessel-segmentation.git
cd blood-vessel-segmentation
```

This repository already contains the model already trained and ready to perform inference. If you want to change any parameter (see the ``train.py`` file) and perform the training again, follow the instructions below. Otherwise, skip to the next step.
## Training  
- To run the training, simply use the following command:
```
python train.py
```

# Inference

- If you are using Windows:
```
cd inference
mkdir build && cd build
cmake ..
#open TFLiteCheck.sln in Visual Studio.
#Build Release x64
cd Release
.\SegmentBloodVessels path_of_image
```
- If you are using MacOs:
```
cd inference
mkdir build && cd build
cmake ..
make
.\SegmentBloodVessels path_of_image
```


- The following result will be displayed:

<p align="center">
  <img src="images/animation.gif" alt="image" width="1000"/>
</p>

* The ``images`` folder has some examples that you can use for testing.

## Contributing

1. Fork the repository.
2. Create a branch for your changes (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## Contact

- **Email**: colaresmarcelo2022@gmail.com
- **LinkedIn**: [engmarcelocolares](https://www.linkedin.com/in/engmarcelocolares/)

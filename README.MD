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

#  🚀 Get started
Clone the repository
```
git clone https://github.com/colaresm/helmet-detection.git
cd helmet-detection
```

This repository already contains the model already trained and ready to perform inference. If you want to change any parameter (see the ``train.py`` file) and perform the training again, follow the instructions below. Otherwise, skip to the next step.
## Training  
- To run the training, simply use the following command:
```
python train.py
```

# Inference
To perform the inference, perform the following steps:
```
cd inference

mkdir build

cd build

cmake ..

make 

.\SegmentBloodVessels
```


## Contributing

1. Fork the repository.
2. Create a branch for your changes (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## Contact

- **Email**: colaresmarcelo2022@gmail.com
- **LinkedIn**: [engmarcelocolares](https://www.linkedin.com/in/engmarcelocolares/)

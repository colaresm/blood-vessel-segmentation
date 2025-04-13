#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <iostream>
#include <memory>

using namespace tflite;

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <image_path>\n";
        return -1;
    }

    std::string modelPath = "../../models/segment_model.tflite";
    std::string imagePath = argv[1];


    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) return -1;

    ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) return -1;

    if (interpreter->AllocateTensors() != kTfLiteOk) return -1;

    int inputIndex = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(inputIndex)->dims;
    int height = dims->data[1];
    int width = dims->data[2];
    int channels = dims->data[3];

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) return -1;

    cv::Mat imageRGB, imageResized;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    cv::resize(imageRGB, imageResized, cv::Size(width, height));
    imageResized.convertTo(imageResized, CV_32FC3, 1.0 / 255.0);

    float* inputTensor = interpreter->typed_tensor<float>(inputIndex);
    std::memcpy(inputTensor, imageResized.data, width * height * channels * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk) return -1;

    int outputIndex = interpreter->outputs()[0];
    TfLiteTensor* output = interpreter->tensor(outputIndex);
    float* outputData = interpreter->typed_tensor<float>(outputIndex);

    int outHeight = output->dims->data[1];
    int outWidth = output->dims->data[2];

    cv::Mat mask(outHeight, outWidth, CV_32FC1, outputData);
    cv::resize(mask, mask, image.size());
    cv::threshold(mask, mask, 0.5, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1, 255.0);

    cv::Mat maskRGB;
    cv::cvtColor(mask, maskRGB, cv::COLOR_GRAY2BGR);

    int titleHeight = 50;
    int totalHeight = image.rows + titleHeight;
    int totalWidth = image.cols * 2;

    cv::Mat result(totalHeight, totalWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::Mat imageToShow;
    cv::cvtColor(imageRGB, imageToShow, cv::COLOR_RGB2BGR);
    imageToShow.copyTo(result(cv::Rect(0, titleHeight, image.cols, image.rows)));
    maskRGB.copyTo(result(cv::Rect(image.cols, titleHeight, image.cols, image.rows)));

    std::string title1 = "Image", title2 = "Segmented vessels";
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 1.0;
    int thickness = 2;
    cv::Scalar color(0, 0, 0);
    int baseLine;

    cv::Size size1 = cv::getTextSize(title1, font, scale, thickness, &baseLine);
    cv::Size size2 = cv::getTextSize(title2, font, scale, thickness, &baseLine);

    int x1 = (image.cols - size1.width) / 2;
    int x2 = image.cols + (image.cols - size2.width) / 2;
    int y = (titleHeight + size1.height) / 2;

    cv::putText(result, title1, cv::Point(x1, y), font, scale, color, thickness);
    cv::putText(result, title2, cv::Point(x2, y), font, scale, color, thickness);

    cv::imshow("Segmentation", result);
    cv::waitKey(0);

    return 0;
}

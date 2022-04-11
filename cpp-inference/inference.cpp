#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define RESOURCE_DIR "./resource/"
#define MODEL_FILENAME RESOURCE_DIR "age_gender_model.tflite"
#define IMGPATH RESOURCE_DIR "36_0_0_face.jpg"
#define MEAN_AVG 130.509485819935

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


cv::Mat normalization(std::string imageFilepath){
    cv::Mat image = cv::imread(imageFilepath);
    cv::resize(image, image, cv::Size(299, 299));
    image = (image - 128)*(128/MEAN_AVG);
    //cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    image.convertTo(image, CV_32FC1, 1.0 / 255);
    return image;
}

int postproc_argmax(float* gender_output){
    std::vector<int> vec(2);
    for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = gender_output[i];
    }
    return *std::max_element(vec.begin(), vec.end());
}
    
int main(){
    std::chrono::system_clock::time_point  start, end;
    /* read input image data */
    cv::Mat image = normalization(IMGPATH);
    std::cout << "image size: " << image.size << std::endl;

    /* tflite */
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    //printf("=== Pre-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(interpreter.get());
    
    // Set data to input tensor
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, image.reshape(0, 1).data, sizeof(float) * 1 * 299 * 299 * 3);
    
    // Run inference
    printf("\n\n=== Run TFLite inference ===\n");
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    //printf("\n\n=== Post-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(interpreter.get());

    // Get data from output tensor
    start = std::chrono::system_clock::now();

    float* age_output = interpreter->typed_output_tensor<float>(0);
    float* gender_output = interpreter->typed_output_tensor<float>(1); 
    int max_index = postproc_argmax(gender_output);
    std::string gender_ = "F";
    if(max_index < 0.5){
        gender_ = "M";
    }
    std::cout << "pred age: " << (int)(*age_output*100) << "  and pred gender: " << gender_ << std::endl;
    
    end = std::chrono::system_clock::now();
    //double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
    double micro_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    //std::cout << "TFLite Inference Latency: " << elapsed << " ms" << std::endl; 
    std::cout << "TFLite C++ Inference Latency: " << micro_elapsed/1000 << " ms" << std::endl;
    return 0;
}


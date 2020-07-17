#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <algorithm>
#include "tensorrt_engine.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorrtInference;

#define NET_NAME "./example/hfnet/"
#define GRAPH_JSON_FILE(net)    net "net_graph.json"
#define GRAPH_WEIGHTS_FILE(net) net "net_weights.bin"
#define GRAPH_ENGINE_FILE(net)  net "net.engine"
#define INFERENCE_JSON_FILE(net) net "/net_inference.json"

#define SAVE_ENGINE 0
#define FP16_FLAG false

#define BACTCH_SIZE 1
#define CHANNEL_SIZE 3
#define HEIGHT_SIZE 720
#define WIDTH_SIZE 1280

int main()
{
    std::string jsonFileName    = GRAPH_JSON_FILE(NET_NAME);
    std::string weightsFileName = GRAPH_WEIGHTS_FILE(NET_NAME);
    std::string engineFileName  = GRAPH_ENGINE_FILE(NET_NAME);
    std::string inferenceFileName = INFERENCE_JSON_FILE(NET_NAME);
#if SAVE_ENGINE
    // save engine file
    tensorrtEngine engine(jsonFileName, weightsFileName, FP16_FLAG);
    engine.saveEnginePlanFile(engineFileName);
#else
    //engine inference
    std::string jpgFile = "./example/hfnet/gray_test.bmp";
    cv::Mat colorMat = cv::imread(jpgFile.c_str());
    cv::Mat grayMat;
    cv::Mat inputMat( 721, 1281, CV_8UC1, cv::Scalar(0));
    cv::Rect rect(0, 0, 1280, 720);
    cv::cvtColor(colorMat, grayMat, cv::COLOR_BGR2GRAY);
    grayMat.copyTo(inputMat(rect));

    tensorrtEngine engine(inferenceFileName);
    std::map<std::string, void*> inputs;
    inputs["gray_image"] = (void*)inputMat.data;
    engine.prepareData(inputs);
    
    engine.doInference(true);
    // for (int i = 0; i < 10; i++) {
    //     auto start = std::chrono::system_clock::now();
    //     engine.doInference(true);
    //     auto end = std::chrono::system_clock::now();
    //     std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // }
    auto result = engine.getInferenceResult();
#endif
}
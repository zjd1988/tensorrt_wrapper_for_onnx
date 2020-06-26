#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include "tensorrt_engine.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorrtInference;

#define NET_NAME "./example/yolov3/"
#define GRAPH_JSON_FILE(net)    net "net_graph.json"
#define GRAPH_WEIGHTS_FILE(net) net "net_weights.bin"
#define GRAPH_ENGINE_FILE(net)  net "net.engine"
#define SAVE_ENGINE 0
#define FP16_FLAG false

#define BACTCH_SIZE 1
#define CHANNEL_SIZE 3
#define HEIGHT_SIZE 320
#define WIDTH_SIZE 192

#define REGION_SIZE 3780
#define CLASSES_SIZE 80
#define BOXES_SIZE 4

void initInputData(std::map<std::string, void*> hostMemMap)
{
    int size = BACTCH_SIZE * CHANNEL_SIZE * HEIGHT_SIZE * WIDTH_SIZE;
    auto input = (float*)(hostMemMap["images"]);
    for(int i = 0; i < size; i++)
    {
        input[i] = 1.0f;
    }
}
void printOutputData(std::map<std::string, void*> hostMemMap)
{
    auto classes_output = (float*)(hostMemMap["classes"]);
    int start = CLASSES_SIZE;
    int end = CLASSES_SIZE + 10;
    for(int i = start; i < end; i++)
    {
        std::cout << classes_output[i] << std::endl;
    }
    auto boxes_output = (float*)(hostMemMap["boxes"]);
    for(int i = 0; i < 10; i++)
    {
        std::cout << boxes_output[i] << std::endl;
    }    

}

int main()
{
    std::string jsonFileName    = GRAPH_JSON_FILE(NET_NAME);
    std::string weightsFileName = GRAPH_WEIGHTS_FILE(NET_NAME);
    std::string engineFileName  = GRAPH_ENGINE_FILE(NET_NAME);
    // save engine file
#if SAVE_ENGINE
    tensorrtEngine engine(jsonFileName, weightsFileName, FP16_FLAG);
    engine.saveEnginePlanFile(engineFileName);
#else
    //engine inference
    // std::string bmpFile = "gray_test.bmp";
    // cv::Mat colorBmp = cv::imread(bmpFile.c_str());
    // cv::Mat grayBmp;
    // cv::Mat inputBmp( 721, 1281, CV_8UC1, cv::Scalar(0));
    // cv::Rect rect(0, 0, 1280, 720);
    // cv::cvtColor(colorBmp, grayBmp, cv::COLOR_BGR2GRAY);
    // grayBmp.copyTo(inputBmp(rect));
    // cv::Mat inputFloatBmp( 721, 1281, CV_32FC1, cv::Scalar(0));
    // inputBmp.convertTo(inputFloatBmp, CV_32F);
    tensorrtEngine engine(engineFileName);
    auto hostMem = engine.getBindingNamesHostMemMap();
    initInputData(hostMem);
    
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        engine.doInference(true);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    printOutputData(hostMem);
#endif
}
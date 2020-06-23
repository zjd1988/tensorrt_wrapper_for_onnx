#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include "tensorrt_engine.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorrtInference;

#define NET_NAME "./example/lenet"
#define GRAPH_JSON_FILE(net)    net "/net_graph.json"
#define GRAPH_WEIGHTS_FILE(net) net "/net_weights.bin"
#define GRAPH_ENGINE_FILE(net)  net "/net.engine"
#define SAVE_ENGINE 1


void initLenetInputData(std::map<std::string, void*> bufferNameMap)
{

}
void initInputData(std::string netName, std::map<std::string, void*> bufferNameMap)
{
    if(netName.compare("lenet") == 0)
        initLenetInputData(bufferNameMap);
    else
        printf("not support %s!\n", netName.c_str());
}

int main()
{
    std::string jsonFileName    = GRAPH_JSON_FILE(NET_NAME);
    std::string weightsFileName = GRAPH_WEIGHTS_FILE(NET_NAME);
    std::string engineFileName  = GRAPH_ENGINE_FILE(NET_NAME);
    // save engine file
#ifdef SAVE_ENGINE
    tensorrtEngine engine(jsonFileName, weightsFileName);
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
    auto bindingNamesMap = engine.getBindingNamesIndexMap();
    initInputData(NET_NAME, bindingNamesMap);
    
    std::vector<void*> data(bindingNamesIndexMap.size());
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        engine.doInference(true);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    getBindingNamesBufferMap["output"]
#endif
    std::cout << "test weights and graph parser !!!" << std::endl;
}
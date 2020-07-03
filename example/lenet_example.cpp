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
#define SAVE_ENGINE 0
#define FP16_FLAG false


#define BACTCH_SIZE 1
#define CHANNEL_SIZE 1
#define HEIGHT_SIZE 32
#define WIDTH_SIZE 32

std::map<int, unsigned char*> initInputData(std::map<std::string, int> hostMemIndexMap, unsigned char* data)
{
    std::map<int, unsigned char*> inputs;
    auto inputIndex = hostMemIndexMap["input"];
    inputs[inputIndex] = data;
    for(int i = 0; i < BACTCH_SIZE * CHANNEL_SIZE * HEIGHT_SIZE * WIDTH_SIZE; i++)
    {
        data[i] = 1;
    }
    return inputs;
}
void printOutputData(std::map<std::string, void*> hostMemMap)
{
    auto output = (float*)(hostMemMap["output"]);
    for(int i = 0; i < 10; i++)
    {
        std::cout << output[i] << std::endl;
    }
}

int main()
{
    std::string jsonFileName    = GRAPH_JSON_FILE(NET_NAME);
    std::string weightsFileName = GRAPH_WEIGHTS_FILE(NET_NAME);
    std::string engineFileName  = GRAPH_ENGINE_FILE(NET_NAME);
    
#if SAVE_ENGINE
    // save engine file
    tensorrtEngine engine(jsonFileName, weightsFileName, FP16_FLAG);
    engine.saveEnginePlanFile(engineFileName);
#else
    //engine inference
    unsigned char* data = (unsigned char*)malloc(BACTCH_SIZE * CHANNEL_SIZE * HEIGHT_SIZE * WIDTH_SIZE);
    tensorrtEngine engine(engineFileName);
    auto hostMemIndex = engine.getBindingNamesIndexMap();
    auto inputs = initInputData(hostMemIndex, data);
    engine.prepareData(inputs);
    
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        engine.doInference(true);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    auto result = engine.getInferenceResult();
    free(data);
    printOutputData(result);
#endif

}
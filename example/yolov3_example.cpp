#include "json/json.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <algorithm>
#include "tensorrt_engine.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace TENSORRT_WRAPPER;

#define NET_NAME "./example/yolov3/"
#define GRAPH_JSON_FILE(net)    net "net_graph.json"
#define GRAPH_WEIGHTS_FILE(net) net "net_weights.bin"
#define GRAPH_ENGINE_FILE(net)  net "net.engine"
#define INFERENCE_JSON_FILE(net) net "net_inference.json"
#define SAVE_DETECT_RESULT(net) net "detect_result.jpg"
#define SAVE_ENGINE 0
#define FP16_FLAG false

#define NMS_THRESH 0.3
#define IOU_THRESH 0.6
#define BACTCH_SIZE 1
#define CHANNEL_SIZE 3
#define HEIGHT_SIZE 512
#define WIDTH_SIZE 384

#define REGION_SIZE 2880
#define CLASSES_SIZE 80
#define BOXES_SIZE 4

template <typename T>
vector<size_t> sort_indexes_e(vector<T> &v)
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });
    return idx;
}

std::map<int, unsigned char*> initInputDataMap(std::map<std::string, int> hostMemIndexMap, cv::Mat& mat)
{
    std::map<int, unsigned char*> inputs;
    auto inputIndex = hostMemIndexMap["input"];
    inputs[inputIndex] = mat.data;
    return inputs;
}

std::map<int, std::vector<int>> initInputDataShapeMap(std::map<std::string, int> hostMemIndexMap, cv::Mat& mat)
{
    std::map<int, std::vector<int>> inputs;
    auto inputIndex = hostMemIndexMap["input"];
    int width = mat.cols;
    int height = mat.rows;
    int channels = mat.channels();
    std::vector<int> shape(4);
    shape[0] = 1;
    shape[1] = height;
    shape[2] = width;
    shape[3] = channels;
    inputs[inputIndex] = shape;
    return inputs;
}

void printOutputData(std::map<std::string, void*> hostMemMap)
{
    auto classes_output = (float*)(hostMemMap["classes"]);
    int start = 0;
    int end = start + 10;
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

void nms(cv::Mat mat, std::map<std::string, void*> hostMemMap)
{
    std::vector<int> boxIndex;
    std::vector<int> classIndex;
    std::vector<float> prob;
    float iou_threshold = IOU_THRESH;
    float *classProb = (float*)(hostMemMap["classes"]);
    float *boxes = (float*)(hostMemMap["boxes"]);
    int strideX = WIDTH_SIZE / 32;
    int strideY = HEIGHT_SIZE / 32;
    //1 find all avalible boxes
    for(int i = 0; i < REGION_SIZE; i++)
    {
        float *start = classProb + CLASSES_SIZE*i;
        float *boxesStart = boxes + BOXES_SIZE*i;
        float maxProb = 0.0f;
        int index = -1;
        for(int j = 0; j < CLASSES_SIZE; j++)
        {
            if(start[j] > maxProb)
            {
                maxProb = start[j];
                index = j;
            }
        }
        if(maxProb > NMS_THRESH)
        {
            classIndex.push_back(index);
            boxIndex.push_back(i);
            prob.push_back(maxProb);
            std::cout << "index " << i << std::endl;
            std::cout << boxesStart[0] << " " << boxesStart[1] << " " << boxesStart[2]*strideX << " " << boxesStart[3]*strideY << std::endl;
        }
        float topLeftX = boxesStart[0] - boxesStart[2]*strideX/2;
        float topLeftY = boxesStart[1] - boxesStart[3]*strideY/2;
        float bottomRightX = boxesStart[0] + boxesStart[2]*strideX/2;
        float bottomRightY = boxesStart[1] + boxesStart[3]*strideY/2;
        boxesStart[0] = topLeftX;
        boxesStart[1] = topLeftY;
        boxesStart[2] = bottomRightX;
        boxesStart[3] = bottomRightY;
    }
    
    for(int i = 0; i < boxIndex.size(); i++)
    {
        float *box = boxes + boxIndex[i]*BOXES_SIZE;
        std::cout << box[0] << " " << box[1] << " " << box[2] << " " << box[3] << std::endl;
    }
    // 2 remove redundant boxes
    auto sortIndex = sort_indexes_e<float>(prob);
    int num_to_keep = 0;
    std::vector<int> supressed(boxIndex.size());
    std::vector<int> keep(boxIndex.size());
    std::vector<int> suppressed(boxIndex.size());
    for(int _i = 0; _i < sortIndex.size(); _i++)
    {
        auto i = sortIndex[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = boxIndex[i];
        float *boxesi = boxes + BOXES_SIZE*boxIndex[i];
        auto ix1 = boxesi[0];
        auto iy1 = boxesi[1];
        auto ix2 = boxesi[2];
        auto iy2 = boxesi[3];
        auto iarea = (boxesi[2] - boxesi[0]) * (boxesi[3] - boxesi[1]);

        for (int64_t _j = _i + 1; _j < sortIndex.size(); _j++) {
            auto j = sortIndex[_j];
            float *boxesj = boxes + BOXES_SIZE*boxIndex[j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, boxesj[0]);
            auto yy1 = std::max(iy1, boxesj[1]);
            auto xx2 = std::min(ix2, boxesj[2]);
            auto yy2 = std::min(iy2, boxesj[3]);

            auto w = std::max(0.0f, xx2 - xx1);
            auto h = std::max(0.0f, yy2 - yy1);
            auto inter = w * h;
            auto jarea = (boxesj[2] - boxesj[0]) * (boxesj[3] - boxesj[1]);
            auto ovr = inter / (iarea + jarea - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    //3 save detected boxes results
    for(int i = 0; i < num_to_keep; i++)
    {
        float *currentBoxes = boxes + keep[i] * BOXES_SIZE;
        float topLeftX = currentBoxes[0];
        float topLeftY = currentBoxes[1];
        float width = currentBoxes[2] - currentBoxes[0];
        float height = currentBoxes[3] - currentBoxes[1];
        cv::Rect rect(topLeftX, topLeftY, width, height);
        cv::rectangle(mat, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);
        // printf("%f %f %f %f \n", currentBoxes[0], currentBoxes[1], currentBoxes[2], currentBoxes[3]);
    }
    cv::imwrite(SAVE_DETECT_RESULT(NET_NAME), mat);
    // cv::waitKey(0);
}

void drawDetectBox(cv::Mat mat, std::map<std::string, void*> buffer, bool saveFlag = false)
{
    auto keep = (int*)buffer["nms_number"];
    auto boxes = (float*)buffer["nms_boxes"];
    auto classes = buffer["nms_classes"];
    int num_to_keep = keep[0];
    // save detected boxes results
    printf("keep boxes num is %d \n", num_to_keep);
    for(int i = 1; i <= num_to_keep; i++)
    {
        float *currentBoxes = boxes + keep[i] * BOXES_SIZE;
        float topLeftX = currentBoxes[0];
        float topLeftY = currentBoxes[1];
        float width = currentBoxes[2] - currentBoxes[0];
        float height = currentBoxes[3] - currentBoxes[1];
        cv::Rect rect(topLeftX, topLeftY, width, height);
        printf("index(%d) %f %f %f %f \n", keep[i], currentBoxes[0], currentBoxes[1], currentBoxes[2], currentBoxes[3]);
    }
    if(saveFlag)
    {
        for(int i = 1; i <= num_to_keep; i++)
        {
            float *currentBoxes = boxes + keep[i] * BOXES_SIZE;
            float topLeftX = currentBoxes[0];
            float topLeftY = currentBoxes[1];
            float width = currentBoxes[2] - currentBoxes[0];
            float height = currentBoxes[3] - currentBoxes[1];
            cv::Rect rect(topLeftX, topLeftY, width, height);
            cv::rectangle(mat, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);
        }
        cv::imwrite(SAVE_DETECT_RESULT(NET_NAME), mat);
    }
}

int main()
{
    std::string jsonFileName    = GRAPH_JSON_FILE(NET_NAME);
    std::string weightsFileName = GRAPH_WEIGHTS_FILE(NET_NAME);
    std::string engineFileName  = GRAPH_ENGINE_FILE(NET_NAME);
    std::string inferenceFileName = INFERENCE_JSON_FILE(NET_NAME);
#if SAVE_ENGINE
    // save engine file
    TensorrtEngine engine(jsonFileName, weightsFileName, FP16_FLAG);
    engine.saveEnginePlanFile(engineFileName);
#else
    //engine inference
    std::string jpgFile = "./example/yolov3/bus.jpg";
    cv::Mat colorJpg = cv::imread(jpgFile.c_str());

    TensorrtEngine engine(inferenceFileName);
    std::map<std::string, void*> inputsData;
    inputsData["bgr_image"] = colorJpg.data;

    engine.prepareData(inputsData);
    
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        engine.doInference(true);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        auto result = engine.getInferenceResult();
        drawDetectBox(colorJpg, result);
    }
    auto result = engine.getInferenceResult();
    drawDetectBox(colorJpg, result, true);
    // nms(colorJpg, result);
#endif
}
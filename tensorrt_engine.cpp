#include "tensorrt_engine.hpp"
#include "create_node.hpp"
#include <fstream>
#include <vector>

using namespace std;

namespace tensorrtInference 
{
    tensorrtEngine::tensorrtEngine(std::string jsonFile, std::string weightsFile, bool fp16Flag)
    {
        builder = nullptr;
        cudaEngine = nullptr;
        // runtime = nullptr;
        // context = nullptr;
        // hostMemMap.clear();
        // deviceMemMap.clear();
        // deviceFp16MemMap.clear();
        builder = createInferBuilder(mLogger);
        CHECK_ASSERT(builder != nullptr, "create builder fail!\n");
        weightsAndGraph.reset(new weightsAndGraphParse(jsonFile, weightsFile));
        CHECK_ASSERT((weightsAndGraph.get()->getInitFlag() != false), "init jsonFile and weightsFile fail!!\n");
        createEngine(1, fp16Flag);
    }
    
    tensorrtEngine::tensorrtEngine(std::string engineFile)
    {
        builder = nullptr;
        cudaRuntime.reset(new CUDARuntime(0));
        executionInfo.reset(new executionParse(cudaRuntime.get(), engineFile));
        CHECK_ASSERT((executionInfo.get()->getInitFlag() != false), "init engineFile fail!!\n");
    }

    tensorrtEngine::~tensorrtEngine()
    {
        if(builder != nullptr)
            builder->destroy();
    }
    
    bool tensorrtEngine::saveEnginePlanFile(std::string saveFile)
    {
        IHostMemory* modelStream = nullptr;
        if(cudaEngine == nullptr)
        {
            LOG("please create net engine first!\n");
            return false;
        }
        // Serialize the engine
        modelStream = cudaEngine->serialize();
        std::ofstream plan(saveFile);
        if (!plan)
        {
            LOG("could not open plan engine file\n");
            return false;
        }
        plan.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        if(modelStream != nullptr)
            modelStream->destroy();
        return true;
    }

    void tensorrtEngine::initConstTensors(std::map<std::string, nvinfer1::ITensor*> &tensors, nvinfer1::INetworkDefinition* network)
    {
        auto constWeightTensors = weightsAndGraph.get()->getConstWeightTensorNames();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        auto size = constWeightTensors.size();
        for(int i = 0; i < size; i++)
        {
            if(tensors.count(constWeightTensors[i]))
                continue;
            LOG("create const tensor %s \n", constWeightTensors[i].c_str());
            auto shape = weightsInfo[constWeightTensors[i]].shape;
            CHECK_ASSERT((shape.size() <= 4), "const tensor shape must less than 3!\n");
            int count = 1;
            for(int j = 0; j < shape.size(); j++)
                count *= shape[j];
            
            nvinfer1::DataType dataType = (weightsInfo[constWeightTensors[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            nvinfer1::Weights weights{dataType, weightsInfo[constWeightTensors[i]].data, count};
            nvinfer1::ILayer* constLayer = nullptr;
            nvinfer1::Dims dims = vectorToDims(shape);
            constLayer = network->addConstant(dims, weights);
            CHECK_ASSERT(constLayer, "create const tensor (%s) fail\n");
            tensors[constWeightTensors[i]] = constLayer->getOutput(0);
        }
    }
    void tensorrtEngine::setNetInput(std::map<std::string, nvinfer1::ITensor*> &tensors, nvinfer1::INetworkDefinition* network)
    {
        int channel, height, width;
        auto inputBlobNames = weightsAndGraph.get()->getNetInputBlobNames();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        int size = inputBlobNames.size();
        for(int i = 0; i < size; i++)
        {
            auto shape = weightsInfo[inputBlobNames[i]].shape;
            if(shape.size() != 4 || inputBlobNames[i].compare("") == 0)
            {
                LOG("input blob shape or input blob name error!\n");
            }
            channel = shape[1];
            height = shape[2];
            width = shape[3];
            nvinfer1::DataType dataType = (weightsInfo[inputBlobNames[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            
            nvinfer1::ITensor* data = network->addInput(inputBlobNames[i].c_str(), dataType, nvinfer1::Dims4{1, channel, height, width});
            CHECK_ASSERT(data!=nullptr, "setNetInput fail\n");
            tensors[inputBlobNames[i]] = data;
        }
    }
    void tensorrtEngine::createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        auto topoOrder = weightsAndGraph.get()->getTopoNodeOrder();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        auto nodeInfoMap = weightsAndGraph.get()->getNodeInfoMap();
        std::map<std::string, nvinfer1::ILayer*> netNode;
        for(int i = 0; i < topoOrder.size(); i++)
        {
            std::string nodeName = topoOrder[i];
            LOG("create %s node\n", nodeName.c_str());
            // if(nodeName.compare("prefix/pred/global_head/vlad/Reshape") == 0)
            //     LOG("run here\n");
            auto nodeConfigInfo = nodeInfoMap[nodeName];
            nvinfer1::ILayer* layer = createNode(network, tensors, nodeConfigInfo.get(), weightsInfo);
            layer->setName(nodeName.c_str());
            CHECK_ASSERT(layer != nullptr, "create %s node fail\n", nodeName);
            netNode[nodeName] = layer;
            auto outputs = nodeConfigInfo.get()->getOutputs();
            for(int i = 0; i < outputs.size(); i++)
            {
                tensors[outputs[i]] = layer->getOutput(i);
                nvinfer1::ITensor *tensor = layer->getOutput(i);
                tensor->setName(outputs[i].c_str());
                nvinfer1::Dims dims = layer->getOutput(i)->getDimensions();
                if(dims.nbDims == 4)
                    LOG("tensor %s  shape is %d %d %d %d\n", outputs[i].c_str(), dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
                else if(dims.nbDims == 3)
                    LOG("tensor %s  shape is %d %d %d\n", outputs[i].c_str(), dims.d[0], dims.d[1], dims.d[2]);
                else if(dims.nbDims == 2)
                    LOG("tensor %s  shape is %d %d\n", outputs[i].c_str(), dims.d[0], dims.d[1]);
                else
                    LOG("tensor %s  shape is %d\n", outputs[i].c_str(), dims.d[0]);
            }
        }
    }
    void tensorrtEngine::createEngine(unsigned int maxBatchSize, bool fp16Flag)
    {
        bool ret = true;
        std::map<std::string, nvinfer1::ITensor*> tensors;
        nvinfer1::INetworkDefinition* network = builder->createNetwork();

        //init constant tensors
        initConstTensors(tensors, network);
        
        //set network input tensor
        setNetInput(tensors, network);
        
        //set network backbone 
        createNetBackbone(tensors, network);

        //mark network output
        auto outputTensors = weightsAndGraph.get()->getNetOutputBlobNames();
        for(int i = 0; i < outputTensors.size(); i++)
        {
            nvinfer1::ITensor* tensor = tensors[outputTensors[i]];
            network->markOutput(*tensor);
        }
        
        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 30);
        if(fp16Flag)
        {
            builder->setFp16Mode(fp16Flag);
            LOG("enable fp16!!!!\n");
        }
        cudaEngine = builder->buildCudaEngine(*network);
        CHECK_ASSERT(cudaEngine != nullptr, "createEngine fail!\n");
        LOG("createEngine success!\n");

        // Don't need the network any more
        network->destroy();
    }

    void tensorrtEngine::prepareData(std::map<std::string, void*> dataMap)
    {
        auto allTensors = executionInfo->getTensorsInfo();
        for(auto inputData : dataMap)
        {
            if(allTensors.count(inputData.first) != 0)
            {
                allTensors[inputData.first]->setHost(dataMap[inputData.first]);
            }
        }
    }

    void tensorrtEngine::doInference(bool syncFlag)
    {
        executionInfo->runInference();
        if(syncFlag)
            cudaRuntime->onWaitFinish();
    }
    std::map<std::string, void*> tensorrtEngine::getInferenceResult()
    {
        return executionInfo->getInferenceResult();
    }

}
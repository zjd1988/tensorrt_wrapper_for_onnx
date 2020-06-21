#include "tensorrt_engine.hpp"
#include "create_node.hpp"
#include "nonzero_cuda_impl.hpp"
#include "convert_cuda_impl.hpp"
#include <fstream>
#include <vector>

using namespace std;

namespace tensorrtInference 
{
    tensorrtEngine::tensorrtEngine(std::string jsonFile, std::string weightsFile)
    {
        builder = nullptr;
        cudaEngine = nullptr;
        runtime = nullptr;
        context = nullptr;
        hostMemMap.clear();
        deviceMemMap.clear();
        deviceFp16MemMap.clear();
        builder = createInferBuilder(mLogger);
        CHECK_ASSERT(builder != nullptr, "create builder fail!\n");
        weightsAndGraph.reset(new weightsAndGraphParse(jsonFile, weightsFile));
        CHECK_ASSERT((weightsAndGraph.get()->getInitFlag() != false), "init jsonFile and weightsFile fail!!\n");
        createEngine(1);
    }
    tensorrtEngine::tensorrtEngine(std::string engineFile)
    {
        char *trtModelStream = nullptr;
        size_t size = 0;
        std::ifstream file(engineFile.c_str(), std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            CHECK_ASSERT(trtModelStream, "malloc fail !\n");
            file.read(trtModelStream, size);
            file.close();
        }
        IRuntime* runtime = createInferRuntime(mLogger);
        CHECK_ASSERT(runtime != nullptr, "create runtime fail!\n");
        cudaEngine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        CHECK_ASSERT(cudaEngine != nullptr, "create engine fail!\n");
        context = cudaEngine->createExecutionContext();
        CHECK_ASSERT(context != nullptr, "create context fail!\n");
        bool ret = mallocEngineMem();
        CHECK_ASSERT(ret != true, "malloc engine host/device mem fail!\n");
        delete[] trtModelStream;
    }
    tensorrtEngine::~tensorrtEngine()
    {
        if(builder != nullptr)
            builder->destroy();
        if(context != nullptr)
            context->destroy();
        if(cudaEngine != nullptr)
            cudaEngine->destroy();
        if(runtime != nullptr)
            runtime->destroy();
        for(auto it : hostMemMap){
            free(it.second);
        }
        for(auto it : deviceMemMap){
            cudaFree(it.second);
        }
        for(auto it : deviceFp16MemMap){
            cudaFree(it.second);
        }
        hostMemMap.clear();
        deviceMemMap.clear();
        deviceFp16MemMap.clear();
        cudaStreamDestroy(stream);
    }
    bool tensorrtEngine::mallocEngineMem()
    {
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int nbBindings = engine.getNbBindings();
        auto byteCount = getBindingByteCount();
        for(int i = 0; i < nbBindings; i++)
        {
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            void *buffer = nullptr;
            if(dataType == nvinfer1::DataType::kHALF)
                buffer = malloc(byteCount[i]);
            else
                buffer = malloc(byteCount[i] * 2);
            CHECK_ASSERT(buffer != nullptr, "malloc %s host mem fail\n", engine.getBindingName(i));
            hostMemMap[i] = buffer;

            void *deviceBuffer = nullptr;
            cudaError_t cudastatus;
            if(dataType == nvinfer1::DataType::kHALF)
            {
                cudastatus = cudaMalloc(&deviceBuffer, byteCount[i]);
                CHECK_ASSERT(cudastatus == cudaSuccess, "malloc %s device mem fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
                deviceFp16MemMap[i] = deviceBuffer;
            }

            deviceBuffer = nullptr;
            cudastatus = cudaMalloc(&deviceBuffer, byteCount[i]);
            CHECK_ASSERT(cudastatus == cudaSuccess, "malloc %s device mem fail: %s\n", engine.getBindingName(i),
                        cudaGetErrorString(cudastatus));
            deviceMemMap[i] = deviceBuffer;
        }
        return true;
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
            if(shape.size() == 4 && shape[0] == 1)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = shape[0];
                dims.d[1] = shape[1];
                dims.d[2] = shape[2];
                dims.d[3] = shape[3];
                constLayer = network->addConstant(dims, weights);
            }
            else if(shape.size() == 3)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = shape[0];
                dims.d[2] = shape[1];
                dims.d[3] = shape[2];
                constLayer = network->addConstant(dims, weights);
            }
            else if(shape.size() == 2)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = 1;
                dims.d[2] = shape[0];
                dims.d[3] = shape[1];
                constLayer = network->addConstant(dims, weights);
            }
            else if(shape.size() == 1)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = 1;
                dims.d[2] = 1;
                dims.d[3] = shape[0];
                constLayer = network->addConstant(dims, weights);
            }
            else
                LOG("const tensor shape size is %d (%d %d %d %d), \n", shape.size(), shape[0], shape[1], shape[2], shape[3]);
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
            }
        }
    }
    void tensorrtEngine::createEngine(unsigned int maxBatchSize)
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
        builder->setMaxWorkspaceSize(1 << 20);
        builder->setFp16Mode(true);
        cudaEngine = builder->buildCudaEngine(*network);
        CHECK_ASSERT(cudaEngine != nullptr, "createEngine fail!\n");
        LOG("createEngine success!\n");

        // Don't need the network any more
        network->destroy();
    }
    // std::map<std::string, int> tensorrtEngine::getBindingNamesIndexMap()
    // {
    //     std::map<std::string, int> bindingNamesIndexMap;
    //     if(cudaEngine == nullptr)
    //         LOG("create engine first!\n");
    //     int nbBinding = cudaEngine->getNbBindings();
    //     for(int i = 0; i < nbBinding; i++)
    //     {
    //         std::string tensorName(cudaEngine->getBindingName(i));
    //         bindingNamesIndexMap[tensorName] = i;
    //     }
    //     return bindingNamesIndexMap;
    // }
    std::map<std::string, void*> tensorrtEngine::getBindingNamesBufferMap()
    {
        std::map<std::string, void*> bindingNamesBufferMap;
        if(cudaEngine == nullptr)
            LOG("create engine first!\n");
        int nbBinding = cudaEngine->getNbBindings();
        for(int i = 0; i < nbBinding; i++)
        {
            std::string tensorName(cudaEngine->getBindingName(i));
            bindingNamesBufferMap[tensorName] = hostMemMap[i];
        }
        return bindingNamesBufferMap;
    }
    std::vector<int> tensorrtEngine::getBindingByteCount()
    {
        std::vector<int> byteCount;
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int nbBinding = engine.getNbBindings();
        int batchSize = engine.getMaxBatchSize();
        for(int i = 0; i < nbBinding; i++)
        {
            int totalByteCount = 1;
            int eleSize = 1;
            nvinfer1::Dims dims = engine.getBindingDimensions(i);
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            int eleByteCount = 1;
            if(dataType == nvinfer1::DataType::kFLOAT)
                eleByteCount = 4;
            else if(dataType == nvinfer1::DataType::kHALF)
                eleByteCount = 2;
            else if(dataType == nvinfer1::DataType::kINT32)
                eleByteCount = 4;
            else if(dataType == nvinfer1::DataType::kBOOL || dataType == nvinfer1::DataType::kINT8)
                eleByteCount = 1;
            else
                CHECK_ASSERT(0, "current only support float/half/int32/bool/int8 !\n");
            for(int i = 0; i < dims.nbDims; i++) {
                eleSize *= dims.d[i];
            }
            totalByteCount = batchSize * eleSize * eleByteCount;
            byteCount.push_back(totalByteCount);
        }
        return byteCount;
    }
    void tensorrtEngine::postprocessOutputData()
    {
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int nbBindings = engine.getNbBindings();
        auto bindingByteCount = getBindingByteCount();
        cudaError_t cudastatus;
        for(int i = 0; i < nbBindings; i++)
        {
            if(engine.bindingIsInput(i) == true)
                continue;
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            int bufferIndex = i;
            int count = bindingByteCount[bufferIndex];
            if(dataType == nvinfer1::DataType::kHALF)
            {
                CudaImpl::ConvertFp16ToFp32CudaImpl(deviceFp16MemMap[bufferIndex], count, deviceMemMap[bufferIndex], stream);
                cudastatus = cudaGetLastError();
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s launch fp16->fp32 fail: %s\n", engine.getBindingName(bufferIndex),
                        cudaGetErrorString(cudastatus));
                count *= 2;
            }
            cudastatus = cudaMemcpyAsync(hostMemMap[bufferIndex], deviceMemMap[bufferIndex], count, cudaMemcpyDeviceToHost, stream);
            CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy to host from device fail: %s\n", engine.getBindingName(bufferIndex),
                        cudaGetErrorString(cudastatus));
        }
    }
    void tensorrtEngine::preprocessInputData()
    {
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int nbBindings = engine.getNbBindings();
        auto bindingByteCount = getBindingByteCount();
        cudaError_t cudastatus;
        for(int i = 0; i < nbBindings; i++)
        {
            if(engine.bindingIsInput(i) != true)
                continue;
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            int bufferIndex = i;
            int count = (dataType == nvinfer1::DataType::kHALF) ? bindingByteCount[bufferIndex] * 2: bindingByteCount[bufferIndex];
            cudaError_t cudastatus = cudaMemcpyAsync(deviceMemMap[bufferIndex], hostMemMap[bufferIndex], count, cudaMemcpyHostToDevice, stream);
            CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy frome host to device fail: %s\n", engine.getBindingName(bufferIndex),
                        cudaGetErrorString(cudastatus));
            if(dataType == nvinfer1::DataType::kHALF)
            {
                CudaImpl::ConvertFp32ToFp16CudaImpl(deviceMemMap[bufferIndex], count / 2, deviceFp16MemMap[bufferIndex], stream);
                cudastatus = cudaGetLastError();
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s fp32->fp16 fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
            }
        }
    }
    void tensorrtEngine::doInference(bool syncFlag)
    {
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int batchSize = engine.getMaxBatchSize();
        int nbBindings = engine.getNbBindings();
        preprocessInputData();
        std::vector<void*> bufferArr;
        for(int i = 0; i < nbBindings; i++)
        {
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            if(dataType == nvinfer1::DataType::kHALF)
                bufferArr.push_back(deviceFp16MemMap[i]);
            else
                bufferArr.push_back(deviceMemMap[i]);
        }
        context->enqueue(batchSize, &bufferArr[0], stream, nullptr);
        postprocessOutputData();
        if(syncFlag)
            cudaStreamSynchronize(stream);
    }
}
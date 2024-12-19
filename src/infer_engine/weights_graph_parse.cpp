/********************************************
 * Filename: weights_graph_parse.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include <fstream>
using namespace std;

namespace tensorrtInference
{
    weightsAndGraphParse::weightsAndGraphParse(std::string &jsonFile, std::string &weightsFile, bool fp16Flag)
    {
        ifstream jsonStream;
        jsonStream.open(jsonFile);
        if(!jsonStream.is_open())
        {
            std::cout << "open json file " << jsonFile << " fail!!!" << std::endl;
            return;
        }
        ifstream weightStream;
        weightStream.open(weightsFile, ios::in | ios::binary);
        if(!weightStream.is_open())
        {
            jsonStream.close();
            std::cout << "open weights file " << weightsFile << " fail!!!" << std::endl;
            return;
        }

        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(jsonStream, root, false))
        {
            std::cout << "parse json file " << jsonFile << " fail!!!" << std::endl;
            jsonStream.close();
            weightStream.close();
            return;
        }
        //extract topo node order
        {
            int size = root["topo_order"].size();
            for(int i = 0; i < size; i++)
            {
                std::string nodeName;
                nodeName = root["topo_order"][i].asString();
                topoNodeOrder.push_back(nodeName);
            }
        }
        //extract weight info 
        {
            auto weihtsInfo = root["weights_info"];
            weightInfo nodeWeightInfo;
            for (auto elem : weihtsInfo.getMemberNames())
            {
                if(elem.compare("net_output") != 0)
                {
                    auto offset = weihtsInfo[elem]["offset"].asInt();
                    auto byteCount  = weihtsInfo[elem]["count"].asInt();
                    auto dataType = weihtsInfo[elem]["data_type"].asInt();
                    std::vector<int> shape;
                    int size = weihtsInfo[elem]["tensor_shape"].size();
                    for(int i = 0; i < size; i++)
                    {
                        auto dim = weihtsInfo[elem]["tensor_shape"][i].asInt();
                        shape.push_back(dim);
                    }
                    nodeWeightInfo.byteCount = byteCount;
                    nodeWeightInfo.dataType = dataType;
                    nodeWeightInfo.shape = shape;
                    char* data = nullptr;
                    if(offset != -1)
                    {
                        data = (char*)malloc(byteCount);
                        CHECK_ASSERT(data, "malloc memory fail!!!!\n");
                        weightStream.seekg(offset, ios::beg);
                        weightStream.read(data, byteCount);
                        weightsData[elem] = data;
                    }
                    nodeWeightInfo.data = data;
                    netWeightsInfo[elem] = nodeWeightInfo;
                    if(offset == -1)
                    {
                        inputTensorNames.push_back(elem);
                    }
                }
                else
                {
                    int size = weihtsInfo[elem].size();
                    for(int i = 0; i < size; i++)
                    {
                        std::string tensorName;
                        tensorName = weihtsInfo[elem][i].asString();
                        outputTensorNames.push_back(tensorName);
                    }
                }
            }
        }
        // extra node info 
        initFlag = extractNodeInfo(root["nodes_info"]);
        jsonStream.close();
        weightStream.close();

        //create engine
        builder = nullptr;
        cudaEngine = nullptr;
        builder = nvinfer1::createInferBuilder(mLogger);
        CHECK_ASSERT(builder != nullptr, "create builder fail!\n");
        createEngine(1, fp16Flag);
        return;
    }
    weightsAndGraphParse::~weightsAndGraphParse()
    {
        if(builder != nullptr)
            builder->destroy();
        if(cudaEngine != nullptr)
            cudaEngine->destroy();
        for(auto it : netWeightsInfo)
        {
            if(it.second.data != nullptr)
            {
                free(it.second.data);
                it.second.data = nullptr;
            }
        }
    }

    bool weightsAndGraphParse::extractNodeInfo(Json::Value &root)
    {
        for (auto elem : root.getMemberNames()) {
            if(root[elem]["op_type"].isString())
            {
                auto op_type = root[elem]["op_type"].asString();

                std::shared_ptr<nodeInfo> node;
                auto parseNodeInfoFromJsonFunc = getNodeParseFuncMap(op_type);
                if(parseNodeInfoFromJsonFunc != nullptr)
                {
                    auto curr_node = parseNodeInfoFromJsonFunc(op_type, root[elem]);
                    if(curr_node == nullptr)
                        return false;
                    // curr_node->printNodeInfo();
                    node.reset(curr_node);
                    nodeInfoMap[elem] = node;
                }
                else
                {
                    LOG("current not support %s node type\n", op_type.c_str());
                }
            }
        }
        return true;
    }

    std::vector<std::string> weightsAndGraphParse::getConstWeightTensorNames()
    {
        std::vector<std::string> constTensorNames;
        for(auto it : nodeInfoMap)
        {
            auto nodeType = it.second->getNodeType();
            auto subNodeType = it.second->getSubNodeType();
            if(nodeType.compare("ElementWise") == 0 || nodeType.compare("Gemm") == 0)
            {
                auto inputs = it.second->getInputs();
                int size = inputs.size();
                for(int i = 0; i < size; i++)
                {
                    if(netWeightsInfo.count(inputs[i]))
                    {
                        auto weight = netWeightsInfo[inputs[i]];
                        if(weight.byteCount == 0)
                            continue;
                        else
                            constTensorNames.push_back(inputs[i]);
                    }
                }
            }
        }
        return constTensorNames;
    }

    void weightsAndGraphParse::initConstTensors(std::map<std::string, nvinfer1::ITensor*> &tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        auto constWeightTensors = getConstWeightTensorNames();
        auto size = constWeightTensors.size();
        for(int i = 0; i < size; i++)
        {
            if(tensors.count(constWeightTensors[i]))
                continue;
            LOG("create const tensor %s \n", constWeightTensors[i].c_str());
            auto shape = netWeightsInfo[constWeightTensors[i]].shape;
            CHECK_ASSERT((shape.size() <= 4), "const tensor shape must less than 4!\n");
            int count = 1;
            for(int j = 0; j < shape.size(); j++)
                count *= shape[j];
            
            nvinfer1::DataType dataType = (netWeightsInfo[constWeightTensors[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            nvinfer1::Weights weights{dataType, netWeightsInfo[constWeightTensors[i]].data, count};
            nvinfer1::ILayer* constLayer = nullptr;
            nvinfer1::Dims dims = vectorToDims(shape);
            constLayer = network->addConstant(dims, weights);
            CHECK_ASSERT(constLayer, "create const tensor (%s) fail\n");
            tensors[constWeightTensors[i]] = constLayer->getOutput(0);
        }
    }
    void weightsAndGraphParse::setNetInput(std::map<std::string, nvinfer1::ITensor*> &tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        int channel, height, width;
        int size = inputTensorNames.size();
        for(int i = 0; i < size; i++)
        {
            auto shape = netWeightsInfo[inputTensorNames[i]].shape;
            if(shape.size() != 4 || inputTensorNames[i].compare("") == 0)
            {
                LOG("input blob shape or input blob name error!\n");
            }
            channel = shape[1];
            height = shape[2];
            width = shape[3];
            nvinfer1::DataType dataType = (netWeightsInfo[inputTensorNames[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;

            nvinfer1::ITensor* data = network->addInput(inputTensorNames[i].c_str(), dataType, 
                    nvinfer1::Dims4{1, channel, height, width});
            CHECK_ASSERT(data!=nullptr, "setNetInput fail\n");
            tensors[inputTensorNames[i]] = data;
        }
    }
    void weightsAndGraphParse::createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        std::map<std::string, nvinfer1::ILayer*> netNode;
        for(int i = 0; i < topoNodeOrder.size(); i++)
        {
            std::string nodeName = topoNodeOrder[i];
            LOG("create %s node\n", nodeName.c_str());
            // if(nodeName.compare("prefix/pred/global_head/vlad/Reshape") == 0)
            //     LOG("run here\n");
            auto nodeConfigInfo = nodeInfoMap[nodeName];
            nvinfer1::ILayer* layer = createNode(network, tensors, nodeConfigInfo.get(), netWeightsInfo);
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
    void weightsAndGraphParse::createEngine(unsigned int maxBatchSize, bool fp16Flag)
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
        for(int i = 0; i < outputTensorNames.size(); i++)
        {
            nvinfer1::ITensor* tensor = tensors[outputTensorNames[i]];
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
    bool weightsAndGraphParse::saveEnginePlanFile(std::string saveFile)
    {
        nvinfer1::IHostMemory* modelStream = nullptr;
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
}
/********************************************
 * Filename: weights_graph_parse.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "parser/weight_graph_parser.hpp"
#include "node_create/create_node.hpp"
#include <fstream>
using namespace std;

namespace TENSORRT_WRAPPER
{

    WeightGraphParser::WeightGraphParser(const std::string &json_file, const std::string &weight_file, bool fp16_flag)
    {
        ifstream json_stream;
        json_stream.open(json_file);
        if(!json_stream.is_open())
        {
            std::cout << "open json file " << json_file << " fail!!!" << std::endl;
            return;
        }

        ifstream weight_stream;
        weight_stream.open(weight_file, ios::in | ios::binary);
        if(!weight_stream.is_open())
        {
            json_stream.close();
            std::cout << "open weights file " << weight_file << " fail!!!" << std::endl;
            return;
        }

        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(json_stream, root, false))
        {
            std::cout << "parse json file " << json_file << " fail!!!" << std::endl;
            json_stream.close();
            weight_stream.close();
            return;
        }

        //extract topo node order
        {
            int size = root["topo_order"].size();
            for(int i = 0; i < size; i++)
            {
                std::string node_name;
                node_name = root["topo_order"][i].asString();
                m_topo_node_order.push_back(node_name);
            }
        }

        //extract weight info 
        {
            auto weights_info = root["weights_info"];
            WeightInfo node_weight_info;
            for (auto elem : weights_info.getMemberNames())
            {
                if(elem.compare("net_output") != 0)
                {
                    auto offset = weights_info[elem]["offset"].asInt();
                    auto byte_count  = weights_info[elem]["count"].asInt();
                    auto dataT_type = weights_info[elem]["data_type"].asInt();
                    std::vector<int> shape;
                    int size = weights_info[elem]["tensor_shape"].size();
                    for(int i = 0; i < size; i++)
                    {
                        auto dim = weights_info[elem]["tensor_shape"][i].asInt();
                        shape.push_back(dim);
                    }
                    node_weight_info.byteCount = byte_count;
                    node_weight_info.dataType = dataT_type;
                    node_weight_info.shape = shape;
                    char* data = nullptr;
                    if(offset != -1)
                    {
                        data = (char*)malloc(byte_count);
                        CHECK_ASSERT(data, "malloc memory fail!!!!\n");
                        weight_stream.seekg(offset, ios::beg);
                        weight_stream.read(data, byte_count);
                        m_weights_data[elem] = data;
                    }
                    node_weight_info.data = data;
                    m_net_weights_info[elem] = node_weight_info;
                    if(offset == -1)
                    {
                        m_input_tensor_names.push_back(elem);
                    }
                }
                else
                {
                    int size = weights_info[elem].size();
                    for(int i = 0; i < size; i++)
                    {
                        std::string tensor_name;
                        tensor_name = weights_info[elem][i].asString();
                        m_output_tensor_names.push_back(tensor_name);
                    }
                }
            }
        }

        // extra node info 
        m_init_flag = extractNodeInfo(root["nodes_info"]);
        json_stream.close();
        weight_stream.close();

        //create engine
        m_builder = nullptr;
        m_cuda_engine = nullptr;
        m_builder = nvinfer1::createInferBuilder(m_logger);
        CHECK_ASSERT(m_builder != nullptr, "create m_builder fail!\n");
        createEngine(1, fp16Flag);
        return;
    }

    WeightGraphParser::~WeightGraphParser()
    {
        if(nullptr != m_builder)
            m_builder->destroy();
        if(nullptr != m_cuda_engine)
            m_cuda_engine->destroy();
        for(auto it : m_net_weights_info)
        {
            if(nullptr != it.second.data)
            {
                free(it.second.data);
                it.second.data = nullptr;
            }
        }
    }

    bool WeightGraphParser::extractNodeInfo(Json::Value &root)
    {
        for (auto elem : root.getMemberNames())
        {
            if(root[elem]["op_type"].isString())
            {
                auto op_type = root[elem]["op_type"].asString();
                std::shared_ptr<NodeInfo> node;
                auto parse_func = getNodeParserFunc(op_type);
                if(nullptr != parse_func)
                {
                    auto curr_node = parseNodeInfoFromJsonFunc(op_type, root[elem]);
                    if(nullptr == curr_node)
                        return false;
                    // curr_node->printNodeInfo();
                    node.reset(curr_node);
                    m_node_info_map[elem] = node;
                }
                else
                {
                    LOG("current not support %s node type\n", op_type.c_str());
                }
            }
        }
        return true;
    }

    std::vector<std::string> WeightGraphParser::getConstWeightTensorNames()
    {
        std::vector<std::string> constTensorNames;
        for(auto it : m_node_info_map)
        {
            auto nodeType = it.second->getNodeType();
            auto subNodeType = it.second->getNodeSubType();
            if(nodeType.compare("ElementWise") == 0 || nodeType.compare("Gemm") == 0)
            {
                auto inputs = it.second->getInputs();
                int size = inputs.size();
                for(int i = 0; i < size; i++)
                {
                    if(m_net_weights_info.count(inputs[i]))
                    {
                        auto weight = m_net_weights_info[inputs[i]];
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

    void WeightGraphParser::initConstTensors(std::map<std::string, nvinfer1::ITensor*> &tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        auto constWeightTensors = getConstWeightTensorNames();
        auto size = constWeightTensors.size();
        for(int i = 0; i < size; i++)
        {
            if(tensors.count(constWeightTensors[i]))
                continue;
            LOG("create const tensor %s \n", constWeightTensors[i].c_str());
            auto shape = m_net_weights_info[constWeightTensors[i]].shape;
            CHECK_ASSERT((shape.size() <= 4), "const tensor shape must less than 4!\n");
            int count = 1;
            for(int j = 0; j < shape.size(); j++)
                count *= shape[j];
            
            nvinfer1::DataType dataType = (m_net_weights_info[constWeightTensors[i]].dataType == OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            nvinfer1::Weights weights{dataType, m_net_weights_info[constWeightTensors[i]].data, count};
            nvinfer1::ILayer* constLayer = nullptr;
            nvinfer1::Dims dims = vectorToDims(shape);
            constLayer = network->addConstant(dims, weights);
            CHECK_ASSERT(constLayer, "create const tensor (%s) fail\n");
            tensors[constWeightTensors[i]] = constLayer->getOutput(0);
        }
    }

    void WeightGraphParser::setNetInput(std::map<std::string, nvinfer1::ITensor*> &tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        int channel, height, width;
        int size = m_input_tensor_names.size();
        for(int i = 0; i < size; i++)
        {
            auto shape = m_net_weights_info[m_input_tensor_names[i]].shape;
            if(shape.size() != 4 || m_input_tensor_names[i].compare("") == 0)
            {
                LOG("input blob shape or input blob name error!\n");
            }
            channel = shape[1];
            height = shape[2];
            width = shape[3];
            nvinfer1::DataType dataType = (m_net_weights_info[m_input_tensor_names[i]].dataType == OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;

            nvinfer1::ITensor* data = network->addInput(m_input_tensor_names[i].c_str(), dataType, 
                nvinfer1::Dims4{1, channel, height, width});
            CHECK_ASSERT(data!=nullptr, "setNetInput fail\n");
            tensors[m_input_tensor_names[i]] = data;
        }
    }

    void WeightGraphParser::createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        std::map<std::string, nvinfer1::ILayer*> netNode;
        for(int i = 0; i < m_topo_node_order.size(); i++)
        {
            std::string nodeName = m_topo_node_order[i];
            LOG("create %s node\n", nodeName.c_str());
            // if(nodeName.compare("prefix/pred/global_head/vlad/Reshape") == 0)
            //     LOG("run here\n");
            auto nodeConfigInfo = m_node_info_map[nodeName];
            nvinfer1::ILayer* layer = createNode(network, tensors, nodeConfigInfo.get(), m_net_weights_info);
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

    void WeightGraphParser::createEngine(unsigned int maxBatchSize, bool fp16Flag)
    {
        bool ret = true;
        std::map<std::string, nvinfer1::ITensor*> tensors;
        nvinfer1::INetworkDefinition* network = m_builder->createNetwork();

        //init constant tensors
        initConstTensors(tensors, network);
        //set network input tensor
        setNetInput(tensors, network);
        //set network backbone 
        createNetBackbone(tensors, network);
        //mark network output
        for(int i = 0; i < m_output_tensor_names.size(); i++)
        {
            nvinfer1::ITensor* tensor = tensors[m_output_tensor_names[i]];
            network->markOutput(*tensor);
        }
        // Build engine
        m_builder->setMaxBatchSize(maxBatchSize);
        m_builder->setMaxWorkspaceSize(1 << 30);
        if(fp16Flag)
        {
            m_builder->setFp16Mode(fp16Flag);
            LOG("enable fp16!!!!\n");
        }
        m_cuda_engine = m_builder->buildCudaEngine(*network);
        CHECK_ASSERT(m_cuda_engine != nullptr, "createEngine fail!\n");
        LOG("createEngine success!\n");
        // Don't need the network any more
        network->destroy();
    }

    bool WeightGraphParser::saveEnginePlanFile(std::string saveFile)
    {
        nvinfer1::IHostMemory* modelStream = nullptr;
        if(m_cuda_engine == nullptr)
        {
            LOG("please create net engine first!\n");
            return false;
        }
        // Serialize the engine
        modelStream = m_cuda_engine->serialize();
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

} // namespace TENSORRT_WRAPPER
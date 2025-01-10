/********************************************
 * Filename: graph_parser.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <fstream>
#include "common/utils.hpp"
#include "common/logger.hpp"
#include "parser/graph_parser.hpp"
#include "node/node_factory.hpp"
#include "node_info/node_info_factory.hpp"

namespace TENSORRT_WRAPPER
{

    GraphParser::GraphParser(const std::string& json_file, const std::string& weight_file)
    {
        // open json file
        std::ifstream json_stream;
        json_stream.open(json_file);
        if(!json_stream.is_open())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "open json file {} fail", json_file);
            return;
        }

        // parse json file
        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(json_stream, root, false))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "parse json file {} fail", json_file);
            json_stream.close();
            return;
        }

        // parse graph node topo order / weights info / nodes info/ engine info
        if (false == parseTopoOrder(root) || false == parseWeightsInfo(root) || 
            false == parseNodesInfo(root) || false == parseEngineInfo(root))
            return;

        // close file stream
        json_stream.close();

        // create tensorrt engine
        m_builder = nvinfer1::createInferBuilder(m_logger);
        CHECK_ASSERT(m_builder != nullptr, "create m_builder fail!\n");
        if (false == createEngine(1, fp16Flag))
            return;

        // set m_init_flag 
        m_init_flag = true;
        return;
    }

    GraphParser::~GraphParser()
    {
        if(nullptr != m_builder)
            delete m_builder;
        if(nullptr != m_cuda_engine)
            delete m_cuda_engine;
        for(auto it : m_net_weights_info)
        {
            if(nullptr != it.second.data)
            {
                free(it.second.data);
                it.second.data = nullptr;
            }
        }
    }

    bool GraphParser::parseTopoOrder(const Json::Value& root)
    {
        // check json value contain specific member
        std::string member_key = "topo_order";
        if (!root.isMember(member_key))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "json value not contain {} member", member_key);
            return false;
        }

        // parse topo order
        auto& topo_order_root = root[member_key];
        for(int i = 0; i < topo_order_root.size(); i++)
        {
            std::string node_name;
            node_name = topo_order_root[i].asString();
            m_node_topo_order.push_back(node_name);
        }
        return true;
    }

    bool GraphParser::parseWeightsInfo(const Json::Value& root, const std::string& weight_file)
    {
        // open weight file
        std::ifstream weight_stream;
        weight_stream.open(weight_file, ios::in | ios::binary);
        if(!weight_stream.is_open())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "open weight file {} fail", weight_file);
            return false;
        }
        size_t file_size = file.tellg();
        file.seekg(0, file.beg);

        // check json value contain specific member
        std::string member_key = "weights_info";
        if (!root.isMember(member_key))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "json value not contain {} member", member_key);
            return false;
        }

        // parse weights info
        auto& weights_root = root[member_key];
        for (auto elem : weights_root.getMemberNames())
        {
            if (0 == elem.compare("net_input"))
            {
                WeightInfo weight_info{};
                auto offset = weights_root[elem]["offset"].asInt();
                auto byte_count  = weights_root[elem]["count"].asInt();
                auto data_type = weights_root[elem]["data_type"].asInt();
                std::vector<int> weight_shape;
                int size = weights_root[elem]["tensor_shape"].size();
                for(int i = 0; i < size; i++)
                {
                    auto dim = weights_root[elem]["tensor_shape"][i].asInt();
                    weight_shape.push_back(dim);
                }
                weight_info.byteCount = byte_count;
                weight_info.dataType = data_type;
                weight_info.shape = weight_shape;
                char* weight_data = nullptr;
                if(offset != -1)
                {
                    weight_data = (char*)malloc(byte_count);
                    if (nullptr == weight_data)
                    {
                        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc buffer for weight:{} fail", elem);
                        return false;
                    }
                    weight_stream.seekg(offset, ios::beg);
                    weight_stream.read(weight_data, byte_count);
                    m_weights_map[elem] = weight_data;
                }
                weight_info.data = weight_data;
                m_net_weights_info[elem] = weight_info;
                if(-1 == offset)
                {
                    m_input_names.push_back(elem);
                }
            }
            else if (0 != elem.compare("net_output"))
            {
                int size = weights_root[elem].size();
                for(int i = 0; i < size; i++)
                {
                    std::string tensor_name;
                    tensor_name = weights_root[elem][i].asString();
                    m_output_names.push_back(tensor_name);
                }
            }
            else
            {
                WeightInfo weight_info{};
                auto offset = weights_root[elem]["offset"].asInt();
                auto byte_count  = weights_root[elem]["count"].asInt();
                auto data_type = weights_root[elem]["data_type"].asInt();
                std::vector<int> weight_shape;
                int size = weights_root[elem]["tensor_shape"].size();
                for(int i = 0; i < size; i++)
                {
                    auto dim = weights_root[elem]["tensor_shape"][i].asInt();
                    weight_shape.push_back(dim);
                }
                weight_info.byte_count = byte_count;
                weight_info.data_type = data_type;
                weight_info.shape = weight_shape;
                char* weight_data = nullptr;
                if(-1 != offset)
                {
                    weight_data = (char*)malloc(byte_count);
                    if (nullptr == weight_data)
                    {
                        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc buffer for weight:{} fail", elem);
                        return false;
                    }
                    if (offset >= file_size || offset + byte_count > file_size)
                    {
                        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "weight:{} offset:{} or byte_count:{} is invalid for weight file size:{}", 
                            elem, offset, byte_count, file_size);
                        return false;
                    }
                    weight_stream.seekg(offset, ios::beg);
                    weight_stream.read(weight_data, byte_count);
                    m_weights_map[elem] = weight_data;
                }
                weight_info.data = weight_data;
                m_net_weights_info[elem] = weight_info;
            }
        }
        return true;
    }

    bool GraphParser::parseNodesInfo(const Json::Value& root)
    {
        // check json value contain specific member
        std::string member_key = "nodes_info";
        if (!root.isMember(member_key))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "json value not contain {} member", member_key);
            return false;
        }

        // parse nodes info
        auto& nodes_root = root[member_key];
        for (auto elem : nodes_root.getMemberNames())
        {
            std::shared_ptr<NodeInfo> node(NodeInfoFactory::create(nodes_root[elem]));
            if(nullptr == node.get())
                return false;
            m_node_info_map[elem] = node;
        }
        return true;
    }

    bool GraphParser::parseEngineInfo(const Json::Value& root)
    {
        // check json value contain specific member
        std::string member_key = "engine_info";
        if (!root.isMember(member_key))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "json value not contain {} member", member_key);
            return false;
        }

        // parse engine info
        auto& engine_root = root[member_key];

        return true;
    }

    std::vector<std::string> GraphParser::getConstWeightTensorNames()
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

    void GraphParser::initConstTensors(std::map<std::string, nvinfer1::ITensor*> &tensors, 
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

    void GraphParser::setNetInput(std::map<std::string, nvinfer1::ITensor*> &tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        int channel, height, width;
        int size = m_input_names.size();
        for(int i = 0; i < size; i++)
        {
            auto shape = m_net_weights_info[m_input_names[i]].shape;
            if(shape.size() != 4 || m_input_names[i].compare("") == 0)
            {
                LOG("input blob shape or input blob name error!\n");
            }
            channel = shape[1];
            height = shape[2];
            width = shape[3];
            nvinfer1::DataType dataType = (m_net_weights_info[m_input_names[i]].dataType == OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;

            nvinfer1::ITensor* data = network->addInput(m_input_names[i].c_str(), dataType, 
                nvinfer1::Dims4{1, channel, height, width});
            CHECK_ASSERT(data!=nullptr, "setNetInput fail\n");
            tensors[m_input_names[i]] = data;
        }
    }

    void GraphParser::createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        std::map<std::string, nvinfer1::ILayer*> netNode;
        for(int i = 0; i < m_node_topo_order.size(); i++)
        {
            std::string nodeName = m_node_topo_order[i];
            LOG("create %s node\n", nodeName.c_str());
            auto nodeConfigInfo = m_node_info_map[nodeName];
            nvinfer1::ILayer* layer = NodeFactory::create(network, tensors, nodeConfigInfo.get(), m_net_weights_info);
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

    void GraphParser::createEngine(unsigned int maxBatchSize, bool fp16Flag)
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
        for(int i = 0; i < m_output_names.size(); i++)
        {
            nvinfer1::ITensor* tensor = tensors[m_output_names[i]];
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

    bool GraphParser::saveEngineFile(const std::string save_file)
    {
        nvinfer1::IHostMemory* engine_buffer = nullptr;
        if(nullptr == m_cuda_engine)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "net engine init fail, please check previous log for more info");
            return false;
        }

        // Serialize the engine
        engine_buffer = m_cuda_engine->serialize();
        std::ofstream file(save_file);
        if (!file.is_open())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "open file {} fail", save_file);
            return false;
        }
        file.write(reinterpret_cast<const char*>(engine_buffer->data()), engine_buffer->size());
        if (nullptr != engine_buffer)
            engine_buffer->destroy();
        return true;
    }

} // namespace TENSORRT_WRAPPER
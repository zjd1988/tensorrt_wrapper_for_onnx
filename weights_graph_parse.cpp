#include "weights_graph_parse.hpp"
#include <fstream>
using namespace std;

namespace tensorrtInference
{
    weightsAndGraphParse::weightsAndGraphParse(std::string &jsonFile, std::string &weightsFile)
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
        //get fp16 flag
        {
            fp16Flag = root["fp16_flag"].asBool();
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
        return;
    }
    weightsAndGraphParse::~weightsAndGraphParse()
    {
        // for(auto it : weightsData) {
        //     if(it.second != nullptr)
        //         free(it.second);
        // }
        // weightsData.clear();

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

    std::vector<std::string>& weightsAndGraphParse::getNetInputBlobNames()
    {
        return inputTensorNames;
    }
    std::vector<std::string>& weightsAndGraphParse::getNetOutputBlobNames()
    {
        return outputTensorNames;
    }    
    const std::vector<std::string>& weightsAndGraphParse::getTopoNodeOrder()
    {
        return topoNodeOrder;
    }
    const std::map<std::string, weightInfo>& weightsAndGraphParse::getWeightsInfo()
    {
        return netWeightsInfo;
    }
    const std::map<std::string, std::shared_ptr<nodeInfo>>& weightsAndGraphParse::getNodeInfoMap()
    {
        return nodeInfoMap;
    }
    std::vector<std::string> weightsAndGraphParse::getConstWeightTensorNames()
    {
        std::vector<std::string> constTensorNames;
        for(auto it : nodeInfoMap)
        {
            auto nodeType = it.second->getNodeType();
            auto subNodeType = it.second->getSubNodeType();
            // LOG("node type: %s , sub node type: %s\n", nodeType.c_str(), subNodeType.c_str());
            // if(nodeType.compare("ElementWise") == 0 || subNodeType.compare("Reshape") == 0)
            if(nodeType.compare("ElementWise") == 0)
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
}
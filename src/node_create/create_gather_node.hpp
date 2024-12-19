#ifndef __CREATE_GATHER_NODE_HPP__
#define __CREATE_GATHER_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createGatherNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif //__CREATE_GATHER_NODE_HPP__
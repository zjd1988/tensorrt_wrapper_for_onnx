#ifndef __CREATE_UNSQUEEZE_NODE_HPP__
#define __CREATE_UNSQUEEZE_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createUnsqueezeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
} //tensorrtInference

#endif //__CREATE_UNSQUEEZE_NODE_HPP__
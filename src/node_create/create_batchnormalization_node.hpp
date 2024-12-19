#ifndef __CREATE_BATCHNORMALIZATION_NODE_HPP__
#define __CREATE_BATCHNORMALIZATION_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createBatchNormalizationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
} //tensorrtInference

#endif //__CREATE_BATCHNORMALIZATION_NODE_HPP__
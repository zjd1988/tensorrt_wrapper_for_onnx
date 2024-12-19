#include "common.hpp"

namespace tensorrtInference
{
    bool broadcastTensor(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& t, const int nbDims)
    {
        const nvinfer1::Dims inputDims = t->getDimensions();
        const int nbInputDims = inputDims.nbDims;
        if (nbInputDims < nbDims)
        {
            auto shape = dimsToVector(inputDims);
            std::vector<int> newShape(nbDims);
            int prefix = nbDims - nbInputDims;
            for(int i = 0; i < nbDims; i++)
            {
                if(i < prefix)
                    newShape[i] = 1;
                else
                    newShape[i] = shape[i - prefix];
            }
            auto newDims = vectorToDims(newShape);
            nvinfer1::IShuffleLayer* reshape = network->addShuffle(*t);
            reshape->setReshapeDimensions(newDims);
            t = reshape->getOutput(0);
        }
        return true;
    }
    bool broadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& tensor1, nvinfer1::ITensor*& tensor2)
    {
        const int t1Dims = tensor1->getDimensions().nbDims;
        const int t2Dims = tensor2->getDimensions().nbDims;

        if (t1Dims == t2Dims)
        {
            return true;
        }

        if (t1Dims > t2Dims)
        {
            return broadcastTensor(network, tensor2, t1Dims);
        }
        return broadcastTensor(network, tensor1, t2Dims);
    }
}
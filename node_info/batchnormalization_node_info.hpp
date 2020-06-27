
#ifndef __BATCHNORMALIZATION_NODE_INFO_HPP__
#define __BATCHNORMALIZATION_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class BatchNormalizationNodeInfo : public nodeInfo
    {
    public:
        BatchNormalizationNodeInfo();
        ~BatchNormalizationNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        float getEpsilon() {return epsilon;}
        float getMomentum() {return momentum;}
    private:
        float epsilon;
        float momentum;
    };
} // tensorrtInference
#endif //__BATCHNORMALIZATION_NODE_INFO_HPP__
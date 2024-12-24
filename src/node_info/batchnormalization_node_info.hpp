
#ifndef __BATCHNORMALIZATION_NODE_INFO_HPP__
#define __BATCHNORMALIZATION_NODE_INFO_HPP__

#include "node_info.hpp"

namespace TENSORRT_WRAPPER
{
    class BatchNormalizationNodeInfo : public NodeInfo
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
} // namespace TENSORRT_WRAPPER
#endif //__BATCHNORMALIZATION_NODE_INFO_HPP__
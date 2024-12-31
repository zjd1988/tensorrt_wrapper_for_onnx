/********************************************
 * Filename: batchnormalization_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class BatchNormalizationNodeInfo : public NodeInfo
    {
    public:
        BatchNormalizationNodeInfo();
        ~BatchNormalizationNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        float getEpsilon() {return m_epsilon;}
        float getMomentum() {return m_momentum;}
    private:
        float                          m_epsilon;
        float                          m_momentum;
    };

} // namespace TENSORRT_WRAPPER
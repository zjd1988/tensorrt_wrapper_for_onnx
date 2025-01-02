/********************************************
 * Filename: activation_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ActivationNodeInfo : public NodeInfo
    {
    public:
        ActivationNodeInfo();
        ~ActivationNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(const std::string type, const Json::Value &root) override;
        virtual void printNodeInfo() override;
        float getAlpha() { return m_alpha; }
        float getBeta() { return m_beta; }

    private:
        float                          m_alpha;
        float                          m_beta;
    };

} // namespace TENSORRT_WRAPPER
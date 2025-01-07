/********************************************
 * Filename: gemm_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class GemmNodeInfo : public NodeInfo
    {
    public:
        GemmNodeInfo();
        ~GemmNodeInfo() = default;
        float getAlpha() { return m_alpha; }
        float getBeta() { return m_beta; }
        int getTransA() {return m_transA; }
        int getTransB() { return m_transB; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual bool verifyParsedNodeInfo() override;
        virtual void printNodeAttributeInfo() override;

    private:
        float                          m_alpha;
        float                          m_beta;
        int                            m_transA;
        int                            m_transB;
    };

} // namespace TENSORRT_WRAPPER
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
        ~GemmNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        float getAlpha() { return alpha; }
        float getBeta() { return beta; }
        int getTransA() {return transA; }
        int getTransB() { return transB; }

    private:
        float alpha;
        float beta;
        int transA;
        int transB;
    };

} // namespace TENSORRT_WRAPPER
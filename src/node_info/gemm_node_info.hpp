
#ifndef __GEMM_NODE_INFO_HPP__
#define __GEMM_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class GemmNodeInfo : public nodeInfo
    {
    public:
        GemmNodeInfo();
        ~GemmNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        float getAlpha(){return alpha;}
        float getBeta(){return beta;}
        int getTransA(){return transA;}
        int getTransB(){return transB;}
    private:
        float alpha;
        float beta;
        int transA;
        int transB;
    };
} // tensorrtInference
#endif //__GEMM_NODE_INFO_HPP__
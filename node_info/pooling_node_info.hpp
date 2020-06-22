
#ifndef __POOLING_NODE_INFO_HPP__
#define __POOLING_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class PoolingNodeInfo : public nodeInfo
    {
    public:
        PoolingNodeInfo();
        ~PoolingNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getKernelShape() { return kernelShape; }
        std::vector<int> getPads() { return pads; }
        std::vector<int> getStrides() { return strides; }
        // std::vector<int> getDilations() { return dilations; }
        std::string      getAutoPad() { return auto_pad; }
        bool             getCeilMode() { return (1 == ceil_mode); }
        int              getCountIncludePad() {return count_include_pad;}
        
    private:
        int ceil_mode;
        int count_include_pad;
        std::string auto_pad;
        std::vector<int> kernelShape;
        std::vector<int> pads;
        std::vector<int> strides;
        // std::vector<int> dilations;
    };
} // tensorrtInference
#endif //__POOLING_NODE_INFO_HPP__
/********************************************
 * Filename: pooling_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class PoolingNodeInfo : public NodeInfo
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

} // namespace TENSORRT_WRAPPER
/********************************************
 * Filename: conv2d_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class Conv2dNodeInfo : public NodeInfo
    {
    public:
        Conv2dNodeInfo();
        ~Conv2dNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getGroup() { return group; }
        std::vector<int> getKernelShape() { return kernel_shape; }
        std::vector<int> getPads() { return pads; }
        std::vector<int> getStrides() { return strides; }
        std::vector<int> getDilation() { return dilation; }

    private:
        int group;
        std::vector<int> kernel_shape;
        std::vector<int> pads;
        std::vector<int> strides;
        std::vector<int> dilation;
    };

} // namespace TENSORRT_WRAPPER
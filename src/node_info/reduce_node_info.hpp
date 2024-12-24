/********************************************
 * Filename: reduce_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ReduceNodeInfo : public NodeInfo
    {
    public:
        ReduceNodeInfo();
        ~ReduceNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getAxes() {return axes;}
        bool getKeepdims() { return keepdims == 1; }

    private:
        std::vector<int> axes;
        int keepdims;
    };

} // namespace TENSORRT_WRAPPER
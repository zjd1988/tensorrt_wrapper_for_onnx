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
        ~ReduceNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getAxes() {return m_axes;}
        bool getKeepdims() { return m_keepdims == 1; }

    private:
        std::vector<int>               m_axes;
        int                            m_keepdims;
    };

} // namespace TENSORRT_WRAPPER
/********************************************
 * Filename: reduce_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ReduceNodeInfo : public NodeInfo
    {
    public:
        ReduceNodeInfo();
        ~ReduceNodeInfo() = default;
        std::vector<int> getAxes() {return m_axes;}
        bool getKeepdims() { return m_keepdims == 1; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual void printNodeAttributeInfo() override;

    private:
        std::vector<int>               m_axes;
        int                            m_keepdims;
    };

} // namespace TENSORRT_WRAPPER
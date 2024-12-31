/********************************************
 * Filename: shuffle_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ShuffleNodeInfo : public NodeInfo
    {
    public:
        ShuffleNodeInfo();
        ~ShuffleNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getPerm() { return m_perm; }
        int getAxis() { return m_axis; }

    private:
        std::vector<int>               m_perm;
        int                            m_axis; //Flatten
    };

} // namespace TENSORRT_WRAPPER
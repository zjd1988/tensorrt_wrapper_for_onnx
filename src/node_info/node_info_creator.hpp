/********************************************
 * Filename: node_info_creator.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    /** abstract node info creator */
    class NodeInfoCreator
    {
    public:
        virtual ~NodeInfoCreator() = default;
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const = 0;

    protected:
        NodeInfoCreator() = default;
    };

    const NodeInfoCreator* getNodeInfoCreator(const std::string type);
    bool insertNodeInfoCreator(const std::string type, const NodeInfoCreator* creator);
    void logRegisteredNodeInfoCreator();

} // namespace TENSORRT_WRAPPER
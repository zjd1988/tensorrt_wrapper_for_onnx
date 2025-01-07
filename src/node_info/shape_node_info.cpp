/********************************************
 * Filename: shape_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/shape_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Shape Node
    ShapeNodeInfo::ShapeNodeInfo() : NodeInfo("Shape")
    {
    }

    class ShapeNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new ShapeNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerShapeNodeInfoCreator()
    {
        insertNodeInfoCreator("Shape", new ShapeNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER
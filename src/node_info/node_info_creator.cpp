/********************************************
 * Filename: node_info_creator.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    static std::map<std::string, const NodeInfoCreator*>& getNodeInfoCreatorMap()
    {
        static std::once_flag gInitFlag;
        static std::map<std::string, const NodeInfoCreator*>* gNodeCreatorMap;
        std::call_once(gInitFlag, [&]() {
            gNodeInfoCreatorMap = new std::map<std::string, const NodeInfoCreator*>;
        });
        return *gNodeInfoCreatorMap;
    }

    extern void registerNodeInfoCreator();
    const NodeCreator* getNodeInfoCreator(const std::string type)
    {
        registerNodeInfoCreator();
        auto& creator_map = getNodeInfoCreatorMap();
        auto iter = creator_map.find(type);
        if (iter == creator_map.end())
        {
            return nullptr;
        }
        if (iter->second)
        {
            return iter->second;
        }
        return nullptr;
    }

    bool insertNodeInfoCreator(const std::string type, const NodeCreator* creator)
    {
        auto& creator_map = getNodeInfoCreatorMap();
        if (creator_map.find(type) != creator_map.end())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "insert duplicate {} execution creator", type);
            return false;
        }
        creator_map.insert(std::make_pair(type, creator));
        return true;
    }

    void logRegisteredNodeCreator()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "registered node creator as follows:");
        auto& creator_map = getNodeCreatorMap();
        for (const auto& it : creator_map)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "registered {} creator", it.first);
        }
        return;
    }

} // namespace TENSORRT_WRAPPER
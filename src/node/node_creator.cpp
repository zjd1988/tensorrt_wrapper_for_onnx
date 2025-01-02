#include "node/node_creator.hpp"

namespace TENSORRT_WRAPPER
{

    static std::map<std::string, const NodeCreator*>& getNodeCreatorMap()
    {
        static std::once_flag gInitFlag;
        static std::map<std::string, const NodeCreator*>* gNodeCreatorMap;
        std::call_once(gInitFlag, [&]() {
            gNodeCreatorMap = new std::map<std::string, const NodeCreator*>;
        });
        return *gNodeCreatorMap;
    }

    extern void registerNodeeCreator();
    const NodeCreator* getNodeCreator(const std::string type)
    {
        registerNodeCreator();
        auto& creator_map = getNodeCreatorMap();
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

    bool insertNodeCreator(const std::string type, const NodeCreator* creator)
    {
        auto& creator_map = getNodeCreatorMap();
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
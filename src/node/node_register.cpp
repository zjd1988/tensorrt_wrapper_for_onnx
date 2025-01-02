/********************************************
 * Filename: node_register.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <mutex>
#include "node_create/create_node.h"

namespace TENSORRT_WRAPPER
{

    extern void registerActivationNodeCreator();
    extern void registerBatchNormalizationNodeCreator();
    extern void registerConcatenationNodeCreator();
    extern void registerElementWiseNodeCreator();

    static std::once_flag s_flag;
    void registerNodeCreator()
    {
        std::call_once(s_flag, [&]() {
            registerActivationNodeCreator();
            registerBatchNormalizationNodeCreator();
            registerConcatenationNodeCreator();
            registerElementWiseNodeCreator();
            logRegisteredNodeCreator();
        });
    }

} // namespace TENSORRT_WRAPPER
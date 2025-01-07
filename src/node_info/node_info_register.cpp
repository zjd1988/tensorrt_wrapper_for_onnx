/********************************************
 * Filename: node_register.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <mutex>
#include "node_info/node_creator.h"

namespace TENSORRT_WRAPPER
{

    extern void registerActivationNodeInfoCreator();
    extern void registerBatchNormalizationNodeInfoCreator();
    extern void registerConcatenationNodeInfoCreator();
    extern void registerElementWiseNodeInfoCreator();
    extern void registerGatherNodeInfoCreator();
    extern void registerIdentityNodeInfoCreator();
    extern void registerNonZeroNodeInfoCreator();
    extern void registerPaddingNodeInfoCreator();
    extern void registerPoolingNodeInfoCreator();
    extern void registerReduceNodeInfoCreator();
    extern void registerResizeNodeInfoCreator();
    extern void registerShapeNodeInfoCreator();
    extern void registerShuffleNodeInfoCreator();
    extern void registerSliceNodeInfoCreator();
    extern void registerSoftmaxNodeInfoCreator();
    extern void registerUnaryNodeInfoCreator();
    extern void registerUnsqueezeNodeInfoCreator();

    static std::once_flag s_flag;
    void registerNodeInfoCreator()
    {
        std::call_once(s_flag, [&]() {
            // register node creator
            registerActivationNodeInfoCreator();
            registerBatchNormalizationNodeInfoCreator();
            registerConcatenationNodeInfoCreator();
            registerGatherNodeInfoCreator();
            registerIdentityNodeInfoCreator();
            registerPaddingNodeInfoCreator();
            registerPoolingNodeInfoCreator();
            registerReduceNodeInfoCreator();
            registerResizeNodeInfoCreator();
            registerShapeNodeInfoCreator();
            registerShuffleNodeInfoCreator();
            registerSliceNodeInfoCreator();
            registerSoftmaxNodeInfoCreator();
            registerUnaryNodeInfoCreator();
            registerUnsqueezeNodeInfoCreator();

            logRegisteredNodeInfoCreator();
        });
    }

} // namespace TENSORRT_WRAPPER
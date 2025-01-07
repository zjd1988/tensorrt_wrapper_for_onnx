/********************************************
 * Filename: node_register.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <mutex>
#include "node/node_creator.h"

namespace TENSORRT_WRAPPER
{

    extern void registerActivationNodeCreator();
    extern void registerBatchNormalizationNodeCreator();
    extern void registerConcatenationNodeCreator();
    extern void registerElementWiseNodeCreator();
    extern void registerGatherNodeCreator();
    extern void registerIdentityNodeCreator();
    extern void registerNonZeroNodeCreator();
    extern void registerPaddingNodeCreator();
    extern void registerPoolingNodeCreator();
    extern void registerReduceNodeCreator();
    extern void registerResizeNodeCreator();
    extern void registerShapeNodeCreator();
    extern void registerShuffleNodeCreator();
    extern void registerSliceNodeCreator();
    extern void registerSoftmaxNodeCreator();
    extern void registerUnaryNodeCreator();
    extern void registerUnsqueezeNodeCreator();

    static std::once_flag s_flag;
    void registerNodeCreator()
    {
        std::call_once(s_flag, [&]() {
            // register node creator
            registerActivationNodeCreator();
            registerBatchNormalizationNodeCreator();
            registerConcatenationNodeCreator();
            registerGatherNodeCreator();
            registerIdentityNodeCreator();
            registerPaddingNodeCreator();
            registerPoolingNodeCreator();
            registerReduceNodeCreator();
            registerResizeNodeCreator();
            registerShapeNodeCreator();
            registerShuffleNodeCreator();
            registerSliceNodeCreator();
            registerSoftmaxNodeCreator();
            registerUnaryNodeCreator();
            registerUnsqueezeNodeCreator();

            logRegisteredNodeCreator();
        });
    }

} // namespace TENSORRT_WRAPPER
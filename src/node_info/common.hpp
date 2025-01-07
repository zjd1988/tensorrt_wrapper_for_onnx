/********************************************
 * Filename: common.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "NvInfer.h"

namespace TENSORRT_WRAPPER
{

    bool broadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& tensor1, nvinfer1::ITensor*& tensor2);

} // namespace TENSORRT_WRAPPER

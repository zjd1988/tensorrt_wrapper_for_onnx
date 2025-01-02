/********************************************
 * Filename: node_creator.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <map>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common/logger.hpp"
#include "parser/graph_parser.hpp"
#include "node/common.hpp"
#include "node/node_info.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    /** abstract node creator */
    class NodeCreator
    {
    public:
        virtual ~NodeCreator() = default;
        virtual nvinfer1::ILayer* onCreate() const = 0;

    protected:
        NodeCreator() = default;
    };

    const NodeCreator* getNodeCreator(const std::string type);
    bool insertNodeCreator(const std::string type, const NodeCreator* creator);
    void logRegisteredNodeCreator();

} // namespace TENSORRT_WRAPPER
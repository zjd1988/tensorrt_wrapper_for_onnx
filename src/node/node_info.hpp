/********************************************
 * Filename: node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "json/json.h"
#include "common/logger.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    class NodeInfo : public NonCopyable
    {
    public:
        NodeInfo() {
            inputs.clear();
            outputs.clear();
        };
        ~NodeInfo() = default;
        void setNodeType(std::string type) { m_type = type; }
        std::string getNodeType() { return m_type; }
        void setNodeSubType(std::string type) { m_sub_type = type; }
        std::string getNodeSubType() { return m_sub_type; }
        std::vector<std::string> getInputs() { return m_inputs; }
        std::vector<std::string> getOutputs() { return m_outputs; }
        void addInput(std::string input) { m_inputs.push_back(input); }
        void addOutput(std::string output) { m_outputs.push_back(output); }
        virtual bool parseNodeInfoFromJson(const std::string type, const Json::Value& root);
        virtual void printNodeInfo();

    private:
        std::string                    m_type;
        std::string                    m_sub_type;
        std::string                    m_name;
        std::vector<std::string>       m_inputs;
        std::vector<std::string>       m_outputs;
    };

} // namespace TENSORRT_WRAPPER
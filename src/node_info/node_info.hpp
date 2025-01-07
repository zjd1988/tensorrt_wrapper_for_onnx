/********************************************
 * Filename: node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "json/json.h"
#include "common/logger.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    // 通用模板函数，用于从 Json::Value 中提取值
    template<typename T>
    bool getValue(const Json::Value& value, const std::string key, T& result, bool optional = false, T default_val = {})
    {
        // check contain key
        if (!value.isMember(key))
        {
            if (!optional)
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} not found");
                return false;
            }
            result = default_val;
            return true;
        }

        // check value is convertible
        if (!value[key].isConvertibleTo(Json::ValueType::stringValue))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value is not convertible to the requested type");
            return false;
        }
        return value[key].as<T>();
    }

    // 特化模板函数，用于从 Json::Value 中提取 std::vector<int>
    template<>
    bool getValue(const Json::Value& value, const std::string key, std::vector<int>& result, bool optional = false, 
        std::vector<int> default_val = {})
    {
        result.clear();
        // check contain key
        if (!value.isMember(key))
        {
            if (!optional)
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} not found");
                return false;
            }
            result = default_val;
            return true;
        }

        // check value is array
        if (!value[key].isArray())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value is not array");
            return false;
        }

        // check array item is convertible to int
        for (const Json::Value& item : value[key])
        {
            if (!item.isConvertibleTo(Json::ValueType::intValue))
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value item not convertible to int");
                return false;
            }
        }

        // convert to result
        for (const Json::Value& item : value[key])
        {
            result.push_back(item.asInt());
        }
        return result;
    }

    // 特化模板函数，用于从 Json::Value 中提取 std::vector<std::string>
    template<>
    bool getValue(const Json::Value& value, const std::string key, std::vector<std::string>& result, bool optional = false, 
        std::vector<std::string> default_val = {})
    {
        result.clear();
        // check contain key
        if (!value.isMember(key))
        {
            if (!optional)
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} not found");
                return false;
            }
            result = default_val;
            return true;
        }

        // check value is array
        if (!value[key].isArray())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value is not array");
            return false;
        }

        // check array item is convertible to string
        for (const Json::Value& item : value[key])
        {
            if (!item.isConvertibleTo(Json::ValueType::stringValue))
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value item not convertible to int");
                return false;
            }
        }

        // convert to result
        for (const Json::Value& item : value[key])
        {
            result.push_back(item.asString());
        }
        return true;
    }

    // 特化模板函数，用于从 Json::Value 中提取 std::vector<float>
    template<>
    bool getValue(const Json::Value& value, const std::string key, std::vector<float>& result, bool optional = false, 
        std::vector<float> default_val = {})
    {
        result.clear();
        // check contain key
        if (!value.isMember(key))
        {
            if (!optional)
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} not found");
                return false;
            }
            result = default_val;
            return true;
        }

        // check value is array
        if (!value[key].isArray())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value is not array");
            return false;
        }

        // check array item is convertible to real
        for (const Json::Value& item : value[key])
        {
            if (!item.isConvertibleTo(Json::ValueType::realValue))
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value item not convertible to float");
                return false;
            }
        }

        // convert to result
        for (const Json::Value& item : value[key])
        {
            result.push_back(item.asFloat());
        }
        return result;
    }

    // 特化模板函数，用于从 Json::Value 中提取 std::vector<double>
    template<>
    bool getValue(const Json::Value& value, const std::string key, std::vector<double>& result, bool optional = false, 
        std::vector<double> default_val = {})
    {
        result.clear();
        // check contain key
        if (!value.isMember(key))
        {
            if (!optional)
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} not found");
                return false;
            }
            result = default_val;
            return true;
        }

        // check value is array
        if (!value[key].isArray())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value is not array");
            return false;
        }

        // check array item is convertible to real
        for (const Json::Value& item : value[key])
        {
            if (!item.isConvertibleTo(Json::ValueType::realValue))
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "key:{} value item not convertible to double");
                return false;
            }
        }

        // convert to result
        for (const Json::Value& item : value[key])
        {
            result.push_back(item.asDouble());
        }
        return result;
    }

    class NodeInfo : public NonCopyable
    {
    public:
        NodeInfo(const std::string type) {
            m_type = type;
            m_sub_type = "";
            m_inputs.clear();
            m_outputs.clear();
        };
        ~NodeInfo() = default;
        std::string getNodeType() { return m_type; }
        std::string getNodeSubType() { return m_sub_type; }
        std::vector<std::string> getInputs() { return m_inputs; }
        std::vector<std::string> getOutputs() { return m_outputs; }
        virtual bool parseNodeInfoFromJson(const std::string type, const Json::Value& root);
        virtual void printNodeInfo();

    protected:
        virtual bool parseNodeBaseInfoFromJson(const Json::Value& root);
        virtual bool parseNodeAttributesFromJson(const Json::Value& root);
        virtual bool verifyParsedNodeInfo();
        virtual void printNodeAttributeInfo();

    protected:
        // node base info
        std::string                    m_type;
        std::string                    m_sub_type;
        std::string                    m_name;
        std::vector<std::string>       m_inputs;
        std::vector<std::string>       m_outputs;
    };

} // namespace TENSORRT_WRAPPER
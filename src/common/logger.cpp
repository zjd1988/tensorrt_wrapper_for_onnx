/********************************************
// Filename: logger.cpp
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#include "common/logger.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#define LM_INFER_LOG_FILE_SIZE 10 * 1024 * 1024   // 10MB
#define LM_INFER_LOG_FILE_NUM 3

namespace TENSORRT_WRAPPER
{

        TrtWrapperLogger::TrtWrapperLogger()
        {
            // init logger name and level
            m_logger_name = "LM_INFER";
            m_logger_level = LM_INFER_LOG_LEVEL_INFO;
            // init logger rotate file sink config
            m_logger_file_name = "";
            m_logger_file_size = LM_INFER_LOG_FILE_SIZE;
            m_logger_file_num = LM_INFER_LOG_FILE_NUM;
        }

        TrtWrapperLogger& TrtWrapperLogger::Instance()
        {
            static TrtWrapperLogger log;
            return log;
        }

        void TrtWrapperLogger::initLogger(std::string file_name, int log_level)
        {
            // set log level
            if (log_level != LM_INFER_LOG_LEVEL_TRACE && log_level != LM_INFER_LOG_LEVEL_DEBUG && 
                log_level != LM_INFER_LOG_LEVEL_INFO  && log_level != LM_INFER_LOG_LEVEL_WARN  && 
                log_level != LM_INFER_LOG_LEVEL_ERROR && log_level != LM_INFER_LOG_LEVEL_FATAL && 
                log_level != LM_INFER_LOG_LEVEL_OFF)
            {
                std::cout << "get invalid log level " << log_level << ", will use INFO as default" << std::endl;
                m_logger_level = LM_INFER_LOG_LEVEL_INFO;
            }
            else
                m_logger_level = log_level;

            // set log rotate
            if (!file_name.empty())
                m_logger_file_name = file_name;

            std::cout << "log level: " << m_logger_level << std::endl;
            std::cout << "logger name: " << m_logger_name << std::endl;

            // init console sink
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            if ("" != m_logger_file_name)
            {
                std::cout << "log file path: " << m_logger_file_name << std::endl;
                // init rotate file sink
                auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(m_logger_file_name, 
                    m_logger_file_size, m_logger_file_num);
                // init logger with console and file sink
                m_logger = std::shared_ptr<spdlog::logger>(new spdlog::logger(m_logger_name, {console_sink, file_sink}));
            }
            else
            {
                // init logger with console
                m_logger = std::shared_ptr<spdlog::logger>(new spdlog::logger(m_logger_name, {console_sink}));
            }
            // set log pattern
            m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %z] [%n] [%^---%L---%$] [thread %t] [%g:%# %!] %v");
            // set log level
            m_logger->set_level(static_cast<spdlog::level::level_enum>(m_logger_level));
            spdlog::register_logger(m_logger);
            return;
        }

        void TrtWrapperLogger::stopLogger()
        {
            // stop logger
            if (nullptr != m_logger.get())
            {
                spdlog::drop(m_logger_name.c_str());
                m_logger.reset();
            }
            return;
        }

        TrtWrapperLogger::~TrtWrapperLogger()
        {
            stopLogger();
        }

        void TrtWrapperLogger::setLevel(int level)
        {
            // update log level
            if (nullptr != m_logger.get())
            {
                m_logger->set_level(static_cast<spdlog::level::level_enum>(level));
            }
            return;
        }

} // namespace TENSORRT_WRAPPER
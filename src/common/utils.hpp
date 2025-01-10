/********************************************
 * Filename: utils.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <assert.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include "common/logger.hpp"

#define CHECK_ASSERT(x, format, args...)                                 \
do {                                                                     \
    if(!(x)) {                                                           \
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_FATAL, format, ##args);    \
        assert(0);                                                       \
    }                                                                    \
} while(0)

#define CUDA_CHECK(_x)                                                   \
    do {                                                                 \
        cudaError_t _err = (_x);                                         \
        if (_err != cudaSuccess) {                                       \
            CHECK_ASSERT(_err, #_x);                                     \
        }                                                                \
    } while (0)

#define CUBLAS_CHECK(_x)                                                 \
do {                                                                     \
    cublasStatus_t _err = (_x);                                          \
    if (_err != CUBLAS_STATUS_SUCCESS) {                                 \
        CHECK_ASSERT(_err, #_x);                                         \
    }                                                                    \
} while (0)

#define CUSOLVER_CHECK(_x)                                               \
do {                                                                     \
    cusolverStatus_t _err = (_x);                                        \
    if (_err != CUSOLVER_STATUS_SUCCESS) {                               \
        CHECK_ASSERT(_err, #_x);                                         \
    }                                                                    \
} while (0)

#define AFTER_KERNEL_LAUNCH()                                            \
do {                                                                     \
    CUDA_CHECK(cudaGetLastError());                                      \
} while (0)

namespace TENSORRT_WRAPPER
{

    class Logger : public nvinfer1::ILogger
    {
    public:

        Logger(): Logger(Severity::kWARNING) {}

        Logger(Severity severity): m_reportable_severity(severity) {}

        void log(Severity severity, const char* msg) override
        {
            // suppress messages with severity enum value greater than the reportable
            if (severity > m_reportable_severity)
                return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR:
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "INTERNAL_ERROR: {}", msg);
                    break;
                }
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "ERROR: {}", msg);
                    break;
                }
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_WARN, "WARNING: {}", msg);
                    break;
                }
                case Severity::kINFO: std::cerr << "INFO: "; break;
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "INFO: {}", msg);
                    break;
                }
                default:
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "UNKONWN: {}", msg);
                    break;
                }
            }
        }

    private:
        Severity m_reportable_severity{Severity::kWARNING};
    };

    typedef enum OnnxDataType
    {
        UNDEFINED,
        FLOAT,
        UINT8,
        INT8,
        UINT16,
        INT16,
        INT32,
        INT64,
        STRING,
        BOOL,
        FLOAT16,
        DOUBLE,
        UINT32,
        UINT64,
        COMPLEX64,
        COMPLEX128,
        BFLOAT16,
    };

    int getOnnxDataTypeByteSize(int type);
    std::vector<float> parseFloatArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    std::vector<int> parseIntArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    int getTensorrtDataType(OnnxDataType onnxDataType);
    std::vector<int> dimsToVector(nvinfer1::Dims dims);
    nvinfer1::Dims vectorToDims(std::vector<int> shape);

} // namespace TENSORRT_WRAPPER

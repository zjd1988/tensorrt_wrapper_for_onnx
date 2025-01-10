/********************************************
 * Filename: utils.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"

namespace TENSORRT_WRAPPER
{

    int getOnnxDataTypeByteSize(int type)
    {
        OnnxDataType data_type = (OnnxDataType)type;
        int byte_size = 0;
        switch (data_type)
        {
            case FLOAT:
            case INT32:
            case UINT32:
            {
                byte_size = 4;
                break;
            }
            case UINT8:
            case INT8:
            case BOOL:
            case STRING:
            {
                byte_size = 1;
                break;
            }
            case UINT16:
            case INT16:
            case FLOAT16:
            case BFLOAT16:
            {
                byte_size = 2;
                break;
            }
            case INT64:
            case DOUBLE:
            case UINT64:
            case COMPLEX64:
            {
                byte_size = 8;
                break;
            }
            case COMPLEX128:
            {
                byte_size = 16;
                break;
            }
            default:
            {
                CHECK_ASSERT(false, "unsupported data type: {}", type);
            }
        }
        return byte_size;
    }

    std::vector<float> parseFloatArrayValue(int data_type, char* data, int byte_count, const std::vector<int> shape)
    {
        bool support_flag = (int(OnnxDataType::FLOAT) == data_type || int(OnnxDataType::DOUBLE) == data_type);
        CHECK_ASSERT(support_flag , "only support FLOAT and DOUBLE");
        int byte_size = getOnnxDataTypeByteSize(data_type);
        int ele_count = 0;
        for(size_t i = 0; i < shape.size(); i++)
        {
            ele_count = (0 == i) ? shape[i] : ele_count * shape[i];
        }
        CHECK_ASSERT((ele_count * byte_size) == byte_count , "ele_count * byte_size != byte_count");

        std::vector<float> arr_value;
        if(int(OnnxDataType::FLOAT) == data_type)
        {
            float *float_data = (float *)data;
            for(int i = 0; i < ele_count; i++)
            {
                arr_value.push_back(float_data[i]);
            }
        }
        else
        {
            double *double_data = (double *)data;
            for(int i = 0; i < ele_count; i++)
            {
                arr_value.push_back(double_data[i]);
            }
        }
        return arr_value;
    }

    std::vector<int> parseIntArrayValue(int data_type, char* data, int byte_count, const std::vector<int> shape)
    {
        bool support_flag = (int(OnnxDataType::INT32) == data_type || int(OnnxDataType::INT64) == data_type);
        CHECK_ASSERT(support_flag , "only support int32 and int64");
        int byte_size = getOnnxDataTypeByteSize(data_type);
        int ele_count = 0;
        for(size_t i = 0; i < shape.size(); i++)
        {
            ele_count = (0 == i) ? shape[i] : ele_count * shape[i];
        }
        CHECK_ASSERT((ele_count * byte_size) == byte_count , "ele_count * byte_size != byte_count");

        std::vector<int> arr_value;
        if(int(OnnxDataType::INT32) == data_type)
        {
            int* int_data = (int *)data;
            for(int i = 0; i < ele_count; i++)
            {
                arr_value.push_back(int_data[i]);
            }
        }
        else
        {
            int64_t* int64_data = (int64_t *)data;
            for(int i = 0; i < ele_count; i++)
            {
                arr_value.push_back(int64_data[i]);
            }
        }
        return arr_value;
    }

    int getTensorrtDataType(OnnxDataType data_type)
    {
        int tensorrt_type = -1;
        switch(data_type)
        {
            case OnnxDataType::FLOAT:
            {
                tensorrt_type = int(nvinfer1::DataType::kFLOAT);
                break;
            }
            case OnnxDataType::FLOAT16:
            {
                tensorrt_type = int(nvinfer1::DataType::kHALF);
                break;
            }
            case OnnxDataType::INT32:
            {
                tensorrt_type = int(nvinfer1::DataType::kINT32);
                break;
            }
            default:
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "current not support convert onnx type: {}", data_type):
                break;
            }
        }
        return tensorrt_type;
    }

    std::vector<int64_t> dimsToVector(nvinfer1::Dims dims)
    {
        std::vector<int64_t> shape_vec;
        for(int i = 0; i < dims.nbDims; i++)
        {
            shape_vec.push_back(dims.d[i]);
        }
        return shape_vec;
    }

    nvinfer1::Dims vectorToDims(std::vector<int64_t> shape)
    {
        size_t nb_dim = shape.size();
        nvinfer1::Dims dims;
        dims.nbDims = nb_dim;
        for(size_t i = 0; i < nb_dim; i++)
        {
            dims.d[i] = shape[i];
        }
        return dims;
    }

} // namespace TENSORRT_WRAPPER

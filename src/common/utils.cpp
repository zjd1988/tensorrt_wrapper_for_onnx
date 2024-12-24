/********************************************
 * Filename: utils.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"

namespace TENSORRT_WRAPPER
{

    int onnxDataTypeEleCount[] = {0, 4, 1, 1, 2, 2, 4, 8, 0, 1, 2, 8, 4, 8, 8, 16, 2};

    std::vector<float> parseFloatArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape)
    {
        bool supportFlag = (dataType == int(OnnxDataType::FLOAT) || dataType == int(OnnxDataType::DOUBLE));
        CHECK_ASSERT(supportFlag , "only support FLOAT and DOUBLE\n");
        int eleCount = onnxDataTypeEleCount[dataType];
        int size = shape.size();
        int shapeCount = 1;
        std::vector<float> arrValue;
        for(int i = 0; i < size; i++)
        {
            shapeCount *= shape[i];
        }
        CHECK_ASSERT((shapeCount * eleCount) == byteCount , "shapeCount * eleCount not equal to byteCount\n");
        if(dataType == int(OnnxDataType::FLOAT))
        {
            float *floatData = (float *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(floatData[i]);
            }
        }
        else
        {
            double *doubleData = (double *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(doubleData[i]);
            }
        }
        return arrValue;
    }

    std::vector<int> parseIntArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape)
    {
        bool supportFlag = (dataType == int(OnnxDataType::INT32) || dataType == int(OnnxDataType::INT64));
        CHECK_ASSERT(supportFlag , "only support int32 and int64\n");
        int eleCount = onnxDataTypeEleCount[dataType];
        int size = shape.size();
        int shapeCount = 1;
        std::vector<int> arrValue;
        for(int i = 0; i < size; i++)
        {
            shapeCount *= shape[i];
        }
        CHECK_ASSERT((shapeCount * eleCount) == byteCount , "shapeCount * eleCount not equal to byteCount\n");
        if(dataType == int(OnnxDataType::INT32))
        {
            int *intData = (int *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(intData[i]);
            }
        }
        else
        {
            int64_t *int64Data = (int64_t *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(int64Data[i]);
            }
        }
        return arrValue;
    }

    int getTensorrtDataType(OnnxDataType onnxDataType)
    {
        switch(onnxDataType)
        {
            case OnnxDataType::FLOAT:
                return int(nvinfer1::DataType::kFLOAT);
            case OnnxDataType::FLOAT16:
                return int(nvinfer1::DataType::kHALF);
            case OnnxDataType::INT32:
                return int(nvinfer1::DataType::kINT32);
            default:
                return -1;
        }
    }

    std::vector<int> dimsToVector(nvinfer1::Dims dims)
    {
        std::vector<int> shapeVec;
        for(int i = 0; i < dims.nbDims; i++)
        {
            shapeVec.push_back(dims.d[i]);
        }
        return shapeVec;
    }

    nvinfer1::Dims vectorToDims(std::vector<int> shape)
    {
        int size = shape.size();
        nvinfer1::Dims dims;
        dims.nbDims = size;
        for(int i = 0; i < size; i++)
        {
            dims.d[i] = shape[i];
        }
        return dims;
    }

} // namespace TENSORRT_WRAPPER

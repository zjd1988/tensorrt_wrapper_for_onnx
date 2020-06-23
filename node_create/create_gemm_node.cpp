#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_gemm_node.hpp"
#include "gemm_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::MatrixOperation getMatrixOperation(const nvinfer1::ITensor& input, bool transpose) {
        if (input.getDimensions().nbDims == 1)
        {
            return nvinfer1::MatrixOperation::kVECTOR;
        }
        else if (transpose)
        {
            return nvinfer1::MatrixOperation::kTRANSPOSE;
        }
        return nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::ILayer* createGemmNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto gemmNodeInfo = (GemmNodeInfo*)nodeConfInfo;
        auto inputs    = gemmNodeInfo->getInputs();
        auto alpha     = gemmNodeInfo->getAlpha();
        auto beta      = gemmNodeInfo->getBeta();
        auto transA    = gemmNodeInfo->getTransA();
        auto transB    = gemmNodeInfo->getTransB();
        tensorrtInference::weightInfo weightInfoA;
        tensorrtInference::weightInfo weightInfoB;
        tensorrtInference::weightInfo weightInfoC;
        bool weightA = false;
        bool weightB = false;
        bool weightC = false;
        nvinfer1::ITensor* inputTensorA = nullptr;
        nvinfer1::ITensor* inputTensorB = nullptr;
        nvinfer1::ITensor* inputTensorC = nullptr;
        inputTensorA = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(inputTensorA != nullptr, "get gemm input tensor:%d fail,topo order error\n", 0);
        auto dataTypeA = inputTensorA->getType();
        CHECK_ASSERT(dataTypeA != nvinfer1::DataType::kINT32, "gemm node: Int32 tensors are not valid input tensors.\n");
        if(nodeWeightsInfo.count(inputs[0]) != 0)
        {
            weightA = true;
            weightInfoA = nodeWeightsInfo[inputs[0]];
        }

        inputTensorB = (tensors.count(inputs[1]) != 0) ? tensors[inputs[1]] : nullptr;
        CHECK_ASSERT(inputTensorB != nullptr, "get gemm input tensor:%d fail,topo order error\n", 1);
        auto dataTypeB = inputTensorB->getType();
        CHECK_ASSERT(dataTypeB != nvinfer1::DataType::kINT32, "gemm node: Int32 tensors are not valid input tensors.\n");        
        if(nodeWeightsInfo.count(inputs[1]) != 0)
        {
            weightB = true;
            weightInfoB = nodeWeightsInfo[inputs[1]];
        }

        if(inputs.size() == 3)
        {
            inputTensorC = (tensors.count(inputs[2]) != 0) ? tensors[inputs[2]] : nullptr;
            CHECK_ASSERT(inputTensorC != nullptr, "get gemm input tensor:%d fail,topo order error\n", 2);
            if(nodeWeightsInfo.count(inputs[2]) != 0)
            {
                weightC = true;
                weightInfoC = nodeWeightsInfo[inputs[2]];
            }
        }
        
        bool canUseFC = weightA == false && weightB == true && weightC == true && alpha == 1.f && beta == 1.f 
        && inputTensorA->getDimensions().nbDims == 3 && inputTensorB->getDimensions().nbDims == 2 
        && inputTensorC->getDimensions().nbDims == 1;
        if(canUseFC)
        {
            LOG("GEMM: using FC layer instead of MM because all criteria were met.");
        }
        
        nvinfer1::MatrixOperation opA = getMatrixOperation(*inputTensorA, transA);
        nvinfer1::MatrixOperation opB = getMatrixOperation(*inputTensorB, transB);

        nvinfer1::IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(*inputTensorA, opA, *inputTensorB, opB);
        nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

        return matmul;
        // if (alpha != 1.f)
        // {
        //     nvinfer1::IConstantLayer* alphaConstant = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        //     nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        //     broadcastTensors(ctx, alphaConstantTensor, matmulTensor);
        //     nvinfer1::IElementWiseLayer* scaledMatmul = network->addElementWise(*alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        //     matmulTensor = scaledMatmul->getOutput(0);
        // }
        // if (inputs.size() == 3)
        // {
        //     nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);

        //     // Scale C if needed
        //     if (beta != 1.f)
        //     {
        //         nvinfer1::IConstantLayer* betaConstant 
        //             = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        //         nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
        //         broadcastTensors(ctx, betaConstantTensor, biasTensor);
        //         nvinfer1::IElementWiseLayer* scaledBias = ctx->network()->addElementWise(
        //             *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
        //         biasTensor = scaledBias->getOutput(0);
        //     }
        //     // A*B may be lower rank than C in TRT, so need to squeeze C.
        //     broadcastTensors(ctx, matmulTensor, biasTensor);
        //     nvinfer1::IElementWiseLayer* biasAdd = network->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
        //     return biasAdd;
        // }
        // if (alpha != 1.f)
        //     return scaledMatmul;
        // else
        //     return matmul;
    }
}
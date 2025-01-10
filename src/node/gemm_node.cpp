/********************************************
 * Filename: gemm_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node_info/gemm_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    static nvinfer1::MatrixOperation getMatrixOperation(const nvinfer1::ITensor& input, bool transpose)
    {
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
        NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info)
    {
        auto gemmNodeInfo = (GemmNodeInfo*)node_info;
        auto inputs    = gemmNodeInfo->getInputs();
        auto alpha     = gemmNodeInfo->getAlpha();
        auto beta      = gemmNodeInfo->getBeta();
        auto transA    = gemmNodeInfo->getTransA();
        auto transB    = gemmNodeInfo->getTransB();
        WeightInfo weightInfoA;
        WeightInfo weightInfoB;
        WeightInfo weightInfoC;
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
        if(weight_info.count(inputs[0]) != 0)
        {
            weightA = true;
            weightInfoA = weight_info[inputs[0]];
        }

        inputTensorB = (tensors.count(inputs[1]) != 0) ? tensors[inputs[1]] : nullptr;
        CHECK_ASSERT(inputTensorB != nullptr, "get gemm input tensor:%d fail,topo order error\n", 1);
        auto dataTypeB = inputTensorB->getType();
        CHECK_ASSERT(dataTypeB != nvinfer1::DataType::kINT32, "gemm node: Int32 tensors are not valid input tensors.\n");        
        if(weight_info.count(inputs[1]) != 0)
        {
            weightB = true;
            weightInfoB = weight_info[inputs[1]];
        }

        if(inputs.size() == 3)
        {
            inputTensorC = (tensors.count(inputs[2]) != 0) ? tensors[inputs[2]] : nullptr;
            CHECK_ASSERT(inputTensorC != nullptr, "get gemm input tensor:%d fail,topo order error\n", 2);
            if(weight_info.count(inputs[2]) != 0)
            {
                weightC = true;
                weightInfoC = weight_info[inputs[2]];
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

        // if (alpha != 1.f)
        // {
        //     nvinfer1::IConstantLayer* alphaConstant = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        //     nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        //     broadcastTensors(ctx, alphaConstantTensor, matmulTensor);
        //     nvinfer1::IElementWiseLayer* scaledMatmul = network->addElementWise(*alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        //     matmulTensor = scaledMatmul->getOutput(0);
        // }
        if (inputs.size() == 3)
        {
            // nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);
            // // Scale C if needed
            // if (beta != 1.f)
            // {
            //     nvinfer1::IConstantLayer* betaConstant 
            //         = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            //     nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
            //     broadcastTensors(ctx, betaConstantTensor, biasTensor);
            //     nvinfer1::IElementWiseLayer* scaledBias = ctx->network()->addElementWise(
            //         *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
            //     biasTensor = scaledBias->getOutput(0);
            // }
            // A*B may be lower rank than C in TRT, so need to squeeze C.
            broadcastTensors(network, matmulTensor, inputTensorC);
            nvinfer1::IElementWiseLayer* biasAdd = network->addElementWise(*matmulTensor, *inputTensorC, nvinfer1::ElementWiseOperation::kSUM);
            return biasAdd;
        }
        return matmul;
        // if (alpha != 1.f)
        //     return scaledMatmul;
        // else
        //     return matmul;
    }

    class GemmNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info) const override 
        {
            return createGemmNode(network, tensors, node_info, weight_info);
        }
    };

    void registerGemmNodeCreator()
    {
        insertNodeCreator("Gemm", new GemmNodeCreator);
    }

} // namespace TENSORRT_WRAPPER
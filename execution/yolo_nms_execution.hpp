#ifndef __YOLO_NMS_EXECUTION_HPP__
#define __YOLO_NMS_EXECUTION_HPP__
#include "execution.hpp"


namespace tensorrtInference
{
    class YoloNMSExecution : public Execution
    {
    public:
        YoloNMSExecution(CUDARuntime *runtime, std::string subType);
        ~YoloNMSExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
        inline Buffer* getSortIdx() { return sortIdx; }
        inline Buffer* getIdx() { return idx; }
        inline Buffer* getSortProb() { return sortProb; }
        inline Buffer* getProb() { return prob; }
        inline Buffer* getMask() { return mask; }
        inline Buffer* getKeep() { return keep; }
        inline Buffer* getClassIdx() {return classIdx;}
        inline Buffer* getCubBuffer() {return cubBuffer;}
        inline Buffer* getBoxesOut() { return boxesOut;}
        inline Buffer* getMaskRemove() { return maskRemove;}
        inline int getCubBufferSize() {return cubBufferSize;}
        inline void setSortIdx(Buffer* buffer) { sortIdx = buffer; }
        inline void setIdx(Buffer* buffer) { idx = buffer; }
        inline void setSortProb(Buffer* buffer) { sortProb = buffer; }
        inline void setProb(Buffer* buffer) { prob = buffer; }
        inline void setMask(Buffer* buffer) { mask = buffer; }
        inline void setKeep(Buffer* buffer) { keep = buffer; }
        inline void setClassIdx(Buffer* buffer) {classIdx = buffer;}
        inline void setCubBuffer(Buffer* buffer) {cubBuffer = buffer;}
        inline void setBoxesOut(Buffer* buffer) {boxesOut = buffer;}
        inline void setMaskRemove(Buffer* buffer) { maskRemove = buffer;}
        inline int getClassesNum() {return classesNum;}
        inline int getBoxesNum() {return boxesNum;}
        inline void setClassesNum(int num) { classesNum = num;}
        inline void setBoxesNum(int num) { boxesNum = num;}
        inline void setCubBufferSize(int num) {cubBufferSize = num;}
        void callYoloNMSExecutionKernel();
        inline void setImgHeight(int height) { imgHeight = height;}
        inline void setImgWidth(int width) { imgWidth = width;}
        inline int getImgHeight() {return imgHeight;}
        inline int getImgWidth() {return imgWidth;}
        inline void setInputBoxesIndex(int index) {inputBoxesIndex = index;}
        inline int getInputBoxesIndex() { return inputBoxesIndex;}
        void recycleBuffers();
    private:
        Buffer *classIdx;
        Buffer *sortIdx;
        Buffer *sortProb;
        Buffer *idx;
        Buffer *prob;
        Buffer *mask;
        Buffer *keep;
        Buffer *cubBuffer;
        Buffer *boxesOut;
        Buffer *maskRemove;
        int inputBoxesIndex;
        int boxesNum;
        int classesNum;
        int cubBufferSize;
        int imgHeight;
        int imgWidth;
    };
} // namespace tensorrtInference 

#endif
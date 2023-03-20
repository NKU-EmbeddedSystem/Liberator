//
// Created by gxl on 2021/2/1.
//

#ifndef PTGRAPH_GRAPHMETA_CUH
#define PTGRAPH_GRAPHMETA_CUH

#include <string>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thread>
#include "TimeRecord.cuh"
#include "globals.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct PartEdgeListInfo {
    SIZE_TYPE partActiveNodeNums;
    SIZE_TYPE partEdgeNums;
    SIZE_TYPE partStartIndex;
};

using namespace std;
#define OLD_MODEL 0
#define NEW_MODEL1 1
#define NEW_MODEL2 2
#define NEW_MODEL3 3
template<class EdgeType>
class TestMeta {
public:
    ~TestMeta();
};

template<class EdgeType>
TestMeta<EdgeType>::~TestMeta() {

}

template<class EdgeType>
class GraphMeta {
public:
    int model;
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    SIZE_TYPE partOverloadSize;
    EDGE_POINTER_TYPE overloadSize;
    SIZE_TYPE sourceNode = 0;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE *nodePointers;
    EDGE_POINTER_TYPE *nodePointersD;
    EdgeType *edgeArray;
    //special for pr
    SIZE_TYPE *outDegree;
    SIZE_TYPE *degree;
    bool *label;
    double *valuePr;
    SIZE_TYPE *value;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    EDGE_POINTER_TYPE *staticNodePointer;
    EDGE_POINTER_TYPE *activeOverloadNodePointers;//no need for model 7
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    EdgeType *overloadEdgeList;//no need for model 7
    //GPU
    uint *resultD;
    cudaStream_t steamStatic, streamDynamic;
    uint *prefixSumTemp;
    EdgeType *staticEdgeListD;
    EdgeType *overloadEdgeListD;//no need for model 7
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodeListD;
    EDGE_POINTER_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    SIZE_TYPE *outDegreeD;
    // async need two labels
    bool *isActiveD;
    thrust::device_ptr<bool> activeLablingThrust;
    thrust::device_ptr<bool> actStaticLablingThrust;
    thrust::device_ptr<bool> actOverLablingThrust;
    thrust::device_ptr<EDGE_POINTER_TYPE> actOverDegreeThrust;
    bool *isStaticActive;
    bool *isOverloadActive;
    SIZE_TYPE *valueD;
    double *valuePrD;
    double *sumD;
    //double *Diff;
    thrust::device_ptr<double> sumDThrust;
    thrust::device_ptr<double> DiffDThrust;
    //SIZE_TYPE *activeNodeListD;
    //SIZE_TYPE *activeNodeLabelingPrefixD;
    //SIZE_TYPE *overloadLabelingPrefixD;
    EDGE_POINTER_TYPE *activeOverloadNodePointersD;//no need for model 7
    EDGE_POINTER_TYPE *activeOverloadDegreeD;//no need for model 7
    double adviseRate;
    int paramSize;
    ALG_TYPE algType;

    void readDataFromFile(const string &fileName, bool isPagerank);

    void transFileUintToUlong(const string &fileName);

    ~GraphMeta();

    void setPrestoreRatio(double adviseK, int paramSize) {
        this->adviseRate = adviseK;
        this->paramSize = paramSize;
    }

    void initGraphHost();

    void initGraphDevice();

    void refreshLabelAndValue();

    void initAndSetStaticNodePointers();

    void setAlgType(ALG_TYPE type) {
        algType = type;
    }

    void setSourceNode(SIZE_TYPE sourceNode) {
        this->sourceNode = sourceNode;
    }

    void fillEdgeArrByMultiThread(uint overloadNodeSize);

    void caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum);
    void checkNode(SIZE_TYPE node);
    void checkNodeforPR(SIZE_TYPE node);
    void checkNodeforSSSP(SIZE_TYPE node);
    void writevalue(string filename);
    void setmodel(int _model){
        this->model = _model;
    }
    bool checkgraph();
    SIZE_TYPE ret_max_partition_size(){
        return max_partition_size;
    }
private:
    SIZE_TYPE max_partition_size;
    SIZE_TYPE max_static_node;
    SIZE_TYPE total_gpu_size;
    uint fragmentSize = 4096;

    void getMaxPartitionSize();

    void initLableAndValue();
    
};

template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromFile(const string &fileName, bool isPagerank) {
    cout << "readDataFromFile" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char *) &this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    if (isPagerank) {
        outDegree = new SIZE_TYPE [vertexArrSize];
        infile.read((char *) outDegree, sizeof(uint) * vertexArrSize);
    }
    if(model==7) {
        //gpuErrorcheck(cudaMallocHost(&nodePointers, sizeof(EdgeType)*vertexArrSize));
        //gpuErrorcheck(cudaMallocManaged(&nodePointers,sizeof(EDGE_POINTER_TYPE)*vertexArrSize));
        //gpuErrorcheck(cudaMemAdvise(nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemAdviseSetAccessedBy,0));
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType)*edgeArrSize));
        //gpuErrorcheck(cudaMallocManaged(&edgeArray,sizeof(EdgeType)*edgeArrSize));
        //gpuErrorcheck(cudaMemAdvise(edgeArray,edgeArrSize*sizeof(EdgeType),cudaMemAdviseSetAccessedBy,0));
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    else{
        nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
        infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
        edgeArray = new EdgeType[edgeArrSize];
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    
    infile.close();
    
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::transFileUintToUlong(const string &fileName) {
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(uint));
    infile.read((char *) &this->edgeArrSize, sizeof(uint));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    outDegree = new uint[vertexArrSize];
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(uint) * vertexArrSize);
    if(model==7) {
        gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(EdgeType)*edgeArrSize));
        //gpuErrorcheck(cudaMallocManaged(&edgeArray,sizeof(EdgeType)*edgeArrSize));
        //gpuErrorcheck(cudaMemAdvise(edgeArray,edgeArrSize*sizeof(EdgeType),cudaMemAdviseSetAccessedBy,0));
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    else{
        edgeArray = new EdgeType[edgeArrSize];
        infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    }
    infile.close();
    vector<ulong> transData(edgeArrSize);
    for (int i = 0; i < edgeArrSize; i++) {
        transData[i] = edgeArray[i];
    }

    std::ofstream outfile(fileName.substr(0, fileName.length() - 4) + "lcsr", std::ofstream::binary);

    outfile.write((char *) &vertexArrSize, sizeof(unsigned int));
    outfile.write((char *) &edgeArrSize, sizeof(unsigned int));
    outfile.write((char *) nodePointers, sizeof(unsigned int) * vertexArrSize);
    outfile.write((char *) transData.data(), sizeof(ulong) * edgeArrSize);

    outfile.close();
}
template<class EdgeType>
void GraphMeta<EdgeType>::checkNode(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
    
    SIZE_TYPE _degree = nodePointers[node+1]-nodePointers[node];
    cout<<"degree: "<<_degree<<endl;
 
}

template<class EdgeType>
void GraphMeta<EdgeType>::checkNodeforPR(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
    SIZE_TYPE _degree = outDegree[node];
    cout<<"degree: "<<_degree<<endl;
}
template<class EdgeType>
void GraphMeta<EdgeType>::checkNodeforSSSP(SIZE_TYPE node)
{
    EDGE_POINTER_TYPE pointer=nodePointers[node];
    cout<<"check node "<<node<<endl;
    cout<<"pointer: "<<pointer<<endl;
   
    EdgeWithWeight edge = edgeArray[pointer];
    cout<<"edge to "<<edge.toNode<<" weighted "<<edge.weight<<endl;
    
}
template<class EdgeType>
bool GraphMeta<EdgeType>::checkgraph(){
    bool flag=true;
    cout<<"checkgraph()"<<endl;
    for(uint i=0;i<vertexArrSize;i++){
        if(nodePointers[i]>=edgeArrSize){
            cout<<"pointer error at "<<i<<" with "<<nodePointers[i]<<endl;
            flag=false;
        }
    }
    for(uint i=0;i<edgeArrSize;i++){
        if(edgeArray[i]>=vertexArrSize){
            cout<<"edge error at "<<i<<" with "<<edgeArray[i]<<endl;
            flag=false;
        }
    }
    if(flag)
    cout<<"check graph correct!!!!!!"<<endl;
    return flag;
}
template<class EdgeType>
GraphMeta<EdgeType>::~GraphMeta() {
    if(model==OLD_MODEL){
        delete[] edgeArray;
        delete[] nodePointers;
        cout << "~GraphMeta" << endl;
        return;
    }
    if(model==7){
        cudaFree(edgeArray);
        cudaFree(nodePointers);
        cout << "~GraphMeta" << endl;
        return;
    }
    
    //delete[] outDegree;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initGraphHost() {
    cout << "initGraphHost()" << endl;
    degree = new SIZE_TYPE[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    if(model!=7)
    activeOverloadNodePointers = new EDGE_POINTER_TYPE[vertexArrSize];

    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
        
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    getMaxPartitionSize();
    initLableAndValue();
    if(model==OLD_MODEL){
        overloadEdgeList = (EdgeType *) malloc(overloadSize * sizeof(EdgeType));
    }
    else if(model!=7){
        gpuErrorcheck(cudaMallocManaged(&overloadEdgeList,sizeof(EdgeType)*overloadSize));
        gpuErrorcheck(cudaMemAdvise(overloadEdgeList,overloadSize*sizeof(EdgeType),cudaMemAdviseSetAccessedBy,0));
        //gpuErrorcheck(cudaMemAdvise(overloadEdgeList,overloadSize*sizeof(EdgeType),cudaMemAdviseSetReadMostly,0));
    }
    
    staticNodePointer = new EDGE_POINTER_TYPE[max_static_node+1];
    for (uint i = 0; i < max_static_node+1; i++) {
        staticNodePointer[i] = nodePointers[i];
    }
}


template<class EdgeType>
void GraphMeta<EdgeType>::initGraphDevice() {
    cout << "initGraphDevice()" << endl;
    
    cudaMalloc(&resultD, grid.x * sizeof(uint));
    cudaMalloc(&prefixSumTemp, vertexArrSize * sizeof(uint));
    //uint* tempResult = new uint[grid.x];
    //memset(tempResult, 0, sizeof(int) * grid.x);
    //cudaMemcpy(resultD, tempResult, grid.x * sizeof(int), cudaMemcpyHostToDevice);

    gpuErrorcheck(cudaPeekAtLastError());
    //cudaMemset(resultD, 0, grid.x * sizeof(uint));

    gpuErrorcheck(cudaStreamCreate(&steamStatic));
    gpuErrorcheck(cudaStreamCreate(&streamDynamic));
    //pre store
    TimeRecord<chrono::milliseconds> totalProcess("pre move data");
    totalProcess.startRecord();
    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(EdgeType)));
    gpuErrorcheck(cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(EdgeType), cudaMemcpyHostToDevice));
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();

    cudaMalloc(&isInStaticD, vertexArrSize * sizeof(bool));
    cudaMalloc(&overloadNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
   
    cudaMalloc(&staticNodePointerD, (max_static_node+1) * sizeof(EDGE_POINTER_TYPE));
    gpuErrorcheck(cudaMemcpy(staticNodePointerD, staticNodePointer, (max_static_node+1) * sizeof(EDGE_POINTER_TYPE), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaPeekAtLastError());
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    //test new model
    if(model==OLD_MODEL)
    cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(EdgeType));
    else{
        gpuErrorcheck(cudaMalloc(&nodePointersD, vertexArrSize*sizeof(EDGE_POINTER_TYPE)));
        cudaMemcpy(nodePointersD,nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemcpyHostToDevice);
    }
    cudaMalloc(&degreeD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&isActiveD, vertexArrSize * sizeof(bool));
    cudaMalloc(&isStaticActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&isOverloadActive, vertexArrSize * sizeof(bool));
    //cudaMalloc(&activeNodeLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
    //cudaMalloc(&overloadLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
 
    //cudaMalloc(&activeNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    if(model!=7){
        cudaMalloc(&activeOverloadNodePointersD, vertexArrSize * sizeof(EDGE_POINTER_TYPE));
        cudaMalloc(&activeOverloadDegreeD, vertexArrSize * sizeof(EDGE_POINTER_TYPE));
    }
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool));
    if(algType == PR) {
        
            cudaMalloc(&outDegreeD, vertexArrSize * sizeof(SIZE_TYPE));
            cudaMemcpy(outDegreeD, outDegree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
            cudaMalloc(&valuePrD, vertexArrSize * sizeof(double));
            cudaMemcpy(valuePrD, valuePr, vertexArrSize * sizeof(double), cudaMemcpyHostToDevice);
            cudaMalloc(&sumD, vertexArrSize * sizeof(double));
            cudaMemset(sumD, 0, vertexArrSize * sizeof(double));
            //cudaMalloc(&Diff,vertexArrSize*sizeof(double));
            //cudaMemset(Diff,0.0,vertexArrSize*sizeof(double));
            sumDThrust = thrust::device_ptr<double>(sumD);
            //DiffDThrust = thrust::device_ptr<double>(Diff);
        
    } else {
        cudaMalloc(&valueD, vertexArrSize * sizeof(SIZE_TYPE));
        cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    }
    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    if(model!=7)
    actOverDegreeThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(activeOverloadDegreeD);
    gpuErrorcheck(cudaPeekAtLastError());
    cout << "initGraphDevice() end" << endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initAndSetStaticNodePointers() {
    staticNodePointer = new EDGE_POINTER_TYPE[vertexArrSize];
    /*memcpy(staticNodePointer, nodePointers, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);*/


}


template<class EdgeType>
void GraphMeta<EdgeType>::getMaxPartitionSize() {
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    unsigned long reduceMem;
    if(algType==PR){
        reduceMem = (paramSize-2) * sizeof(SIZE_TYPE) * (long) vertexArrSize;
        reduceMem += sizeof(double) * 2 * (long)vertexArrSize;
    }
    else
    reduceMem = paramSize * sizeof(SIZE_TYPE) * (long) vertexArrSize + vertexArrSize*sizeof(EDGE_POINTER_TYPE);
    
    cout << "reduceMem " << reduceMem  << " ParamsSize " << paramSize << endl;
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;     
    cout << "available memory for edges "<< (availMemory - reduceMem) << " sizeof EdgeType is "<<sizeof(EdgeType)<<endl;
    total_gpu_size = (availMemory - reduceMem) / sizeof(EdgeType);
    cout<<"total_gpu_size: "<<total_gpu_size<<endl;
    //getchar();
    //float adviseK = (10 - (float) edgeListSize / (float) totalSize) / 9;
    //uint dynamicDataMax = edgeListSize * edgeSize -i
    
        float adviseK = (10 - (double) edgeArrSize / (double) total_gpu_size) / 9;
            cout << "adviseK " << adviseK << endl;
            if (adviseK < 0) {
                adviseK = 0.5;
                cout << "adviseK " << adviseK << endl;
            }
            if (adviseK > 1) {
                adviseK = 1.0;
                cout << "adviseK " << adviseK << endl;
            }
            cout << "adviseRate " << adviseRate << endl;
            if (adviseRate > 0) {
                adviseK = adviseRate;
            }
            if(model!=7){
            max_partition_size = adviseK * total_gpu_size;
            }
            else
            max_partition_size = total_gpu_size;

            if (max_partition_size > edgeArrSize) {
                max_partition_size = edgeArrSize;
                cout<<"GPU fill all the edges!!!"<<endl;
            }
            
            printf("static memory is %ld  max static edge size is %ld\n gpu total edge size %ld \n",
                availMemory - reduceMem,
                max_partition_size, 
                total_gpu_size);
            if (max_partition_size > UINT_MAX) {
                printf("bigger than DIST_INFINITY\n");
                max_partition_size = UINT_MAX;
            }
            //SIZE_TYPE temp = max_partition_size % fragmentSize;
            //max_partition_size = max_partition_size - temp;
            max_static_node = 0;
            SIZE_TYPE edgesInStatic = 0;
            for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
                    isInStatic[i] = true;
                    if (i > max_static_node) max_static_node = i;
                    edgesInStatic += degree[i];
                } else {
                    isInStatic[i] = false;
                }
            }

            //cout << "max_partition_size " << max_partition_size << " nodePointers[vertexArrSize-1]" << nodePointers[vertexArrSize-1] << " edgesInStatic " << edgesInStatic << endl;

            partOverloadSize = total_gpu_size - max_partition_size;
            overloadSize = edgeArrSize - edgesInStatic;

    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << endl;
    cout << " max staticnode is "<<max_static_node<<endl;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initLableAndValue() {

    label = new bool[vertexArrSize];
    if (algType == PR) {
        valuePr = new double[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0;
        }
    } else {
        value = new SIZE_TYPE[vertexArrSize];
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::refreshLabelAndValue() {
    cout << "refreshLabelAndValue()" << endl;
    if (algType == PR) {
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0 / vertexArrSize;
        }
        //cout << "refreshLabelAndValue() end1" << endl;
        cudaMemcpy(valuePrD, valuePr, vertexArrSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        
        //cout << "refreshLabelAndValue() end2" << endl;
        gpuErrorcheck(cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool)));
        //cout << "refreshLabelAndValue() end3" << endl;
    } else {
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize+1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                cout << "sourceNode " << sourceNode << endl;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = vertexArrSize+1;
                }

        }
        cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool)));
    }

    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    actOverDegreeThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(activeOverloadDegreeD);
    
}

template<class EdgeType>
void GraphMeta<EdgeType>::fillEdgeArrByMultiThread(uint overloadNodeSize) {
    //cout << "fillEdgeArrByMultiThread" << endl;
    int threadNum = 20;
    if (overloadNodeSize < 50) {
        threadNum = 1;
    }
    thread runThreads[threadNum];

    for (int threadIndex = 0; threadIndex < threadNum; threadIndex++) {
        //cout << "======= threadIndex " << threadIndex << endl;
        runThreads[threadIndex] = thread([&, threadIndex] {
            float waitToHandleNum = overloadNodeSize;
            float numThreadsF = threadNum;
            unsigned int chunkSize = ceil(waitToHandleNum / numThreadsF);
            unsigned int left, right;
            //cout << "======= threadIndex " << threadIndex << endl;
            left = threadIndex * chunkSize;
            right = min(left + chunkSize, overloadNodeSize);
            unsigned int thisNode;
            unsigned int thisDegree;
            EDGE_POINTER_TYPE fromHere = 0;
            EDGE_POINTER_TYPE fromThere = 0;
            //cout << left << "=======" << right << endl;
            for (unsigned int i = left; i < right; i++) {
                thisNode = overloadNodeList[i];
                thisDegree = degree[thisNode];
                fromHere = activeOverloadNodePointers[i];
                fromThere = nodePointers[thisNode];
                
                // if(activeOverloadNodePointers[i] > overloadSize) {
                //     cout << "activeOverloadNodePointers[" << i << "] is " << activeOverloadNodePointers[i] << endl;
                //     break;
                // }
                for (unsigned int j = 0; j < thisDegree; j++) {
                    overloadEdgeList[fromHere + j] = edgeArray[fromThere + j];
                    //cout << fromHere + j << " : " << overloadEdgeList[fromHere + j] << endl;
                }
                
            }
        });
    }
    for (unsigned int t = 0; t < threadNum; t++) {
        runThreads[t].join();
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum) {
    partEdgeListInfoArr.clear();
    if (partOverloadSize < overloadEdgeNum) {
        uint left = 0;
        uint right = overloadNodeNum - 1;
        while ((activeOverloadNodePointers[right] + degree[overloadNodeList[right]] -
                activeOverloadNodePointers[left]) >
               partOverloadSize) {

            //cout << "left " << left << " right " << right << endl;
            //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

            uint start = left;
            uint end = right;
            uint mid;
            while (start <= end) {
                mid = (start + end) / 2;
                EDGE_POINTER_TYPE headDistance = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
                EDGE_POINTER_TYPE tailDistance =
                        activeOverloadNodePointers[mid] + degree[overloadNodeList[mid]] -
                        activeOverloadNodePointers[left];
                if (headDistance <= partOverloadSize && tailDistance > partOverloadSize) {
                    //cout << "left " << left << " mid " << mid << endl;
                    //cout << "activeOverloadNodePointers[mid] "<< activeOverloadNodePointers[mid] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

                    break;
                } else if (tailDistance <= partOverloadSize) {
                    start = mid + 1;
                } else if (headDistance > partOverloadSize) {
                    end = mid - 1;
                }
            }
            
            PartEdgeListInfo info;
            info.partActiveNodeNums = mid - left;
            info.partEdgeNums = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
            info.partStartIndex = left;
            partEdgeListInfoArr.push_back(info);
            left = mid;
            //cout << "left " << left << " right " << right << endl;
            //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;

        }

        //cout << "left " << left << " right " << right << endl;
        //cout << "activeOverloadNodePointers[right] + degree[overloadNodeList[right]] "<< activeOverloadNodePointers[right] + degree[overloadNodeList[right]] <<" activeOverloadNodePointers[left] " << activeOverloadNodePointers[left] << endl;


        PartEdgeListInfo info;
        info.partActiveNodeNums = right - left + 1;
        info.partEdgeNums =
                activeOverloadNodePointers[right] + degree[overloadNodeList[right]] - activeOverloadNodePointers[left];
        info.partStartIndex = left;
        partEdgeListInfoArr.push_back(info);
    } else {
        PartEdgeListInfo info;
        info.partActiveNodeNums = overloadNodeNum;
        info.partEdgeNums = overloadEdgeNum;
        info.partStartIndex = 0;
        partEdgeListInfoArr.push_back(info);
    }
}
template<class EdgeType>
void GraphMeta<EdgeType>::writevalue(string filename){
    std::ofstream outfile(filename);
    unsigned int num = 0;
    switch (algType){
        case PR:
        cudaMemcpy(valuePr,valuePrD,vertexArrSize*sizeof(double),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        
        for(unsigned long long i=0;i<vertexArrSize;i++){
            if(valuePr[i]==0.15)
            num++;
            outfile << valuePr[i] <<std::endl;
        }
        cout<<"0.15 num: "<<num<<endl;
        break;
        default:
        cudaMemcpy(value,valueD,vertexArrSize*sizeof(SIZE_TYPE),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        gpuErrorcheck(cudaPeekAtLastError());
        for(unsigned long long i=0;i<vertexArrSize;i++){
            outfile << value[i] <<std::endl;
        }
        break;
    }
    outfile.close();
    
}
#endif //PTGRAPH_GRAPHMETA_CUH

#include"globals.cuh"
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
#include "TimeRecord.cuh"
#include"gpu_kernels.cuh"
#pragma once
struct StaticRegionInfo
{
    uint max_node;
    uint max_partion_size;
};
EDGE_POINTER_TYPE vertexArrSize, edgeArrSize;

StaticRegionInfo getMaxPartionSize(int paramSize, unsigned long long edgeArrSize, EDGE_POINTER_TYPE vertexArrSize, EDGE_POINTER_TYPE* nodePointers, uint*degree,
                       bool* isInStatic){
    unsigned long max_partition_size;
    unsigned long max_static_node;
    unsigned long total_gpu_size;
    uint fragmentSize = 4096;
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    long reduceMem;
    reduceMem = 6*sizeof(uint)*(long)vertexArrSize;
    reduceMem += 4 * sizeof(bool) * (long) vertexArrSize;
    reduceMem += vertexArrSize*sizeof(EDGE_POINTER_TYPE);
    cout << "reduceMem " << reduceMem << " testNumNodes " << vertexArrSize << " edgeArrSize " << edgeArrSize << " ParamsSize " << paramSize << endl;
    total_gpu_size = (availMemory - reduceMem) / sizeof(uint);
    max_partition_size = total_gpu_size;
    if (max_partition_size > edgeArrSize) {
        max_partition_size = edgeArrSize;
    }
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    printf("static memory is %ld totalGlobalMem is %ld, max static edge size is %ld\n gpu total edge size %ld \n multiprocessors %d \n",
                availMemory - reduceMem,
                dev.totalGlobalMem, max_partition_size, total_gpu_size, dev.multiProcessorCount);
    if (max_partition_size > UINT_MAX) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = UINT_MAX;
    }
    uint temp = max_partition_size % fragmentSize;
    max_partition_size = max_partition_size - temp;
    max_static_node = 0;
    uint edgesInStatic = 0;
    for (uint i = 0; i < vertexArrSize; i++) {
        if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > max_static_node) max_static_node = i;
                edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }
    StaticRegionInfo info;
    info.max_node = max_static_node;
    info.max_partion_size = max_partition_size;
    return info;
}
void refreshLableAndValue(bool* isActiveD, bool *isStaticActiveD, bool* isoverloadActiveD,uint* value, uint* valueD){
    cudaMemset(isActiveD,1,sizeof(bool)*vertexArrSize);
    cudaMemset(isStaticActiveD,0,sizeof(bool)*vertexArrSize);
    cudaMemset(isoverloadActiveD,0,sizeof(bool)*vertexArrSize);
    cudaMemcpy(valueD,value,vertexArrSize*sizeof(uint),cudaMemcpyHostToDevice);
}

__global__ 
void cc_kernelStatic(uint activeNodesNum, uint *activeNodeListD,
                    uint *staticNodePointerD, uint *degreeD,
                    uint *edgeListD, uint *valueD, bool *isActiveD, bool *isInStaticD) {
    streamVertices(activeNodesNum, [&](uint index) {
        uint id = activeNodeListD[index];
        if (isInStaticD[id]) {
            uint edgeIndex = staticNodePointerD[id];
            uint sourceValue = valueD[id];
            for (uint i = 0; i < degreeD[id]; i++) {
                uint vertexId = edgeListD[edgeIndex + i];
                if (sourceValue < valueD[vertexId]) {
                    atomicMin(&valueD[vertexId], sourceValue);
                    isActiveD[vertexId] = 1;
                }
            }
        }
    });
}

void New_CC_opt(string fileName,int model,int testTimes){
    if(model!=7){
        cout<<"model not match"<<endl;
        return;
    }
    EDGE_POINTER_TYPE *nodePointers;
    uint* edgeArray; 
    
    StaticRegionInfo StaticInfo;
    //ReadDataFile
    cout << "readDataFromFile" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char *) &edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << endl;
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    //gpuErrorcheck(cudaMallocHost(&nodePointers, sizeof(EDGE_POINTER_TYPE)*vertexArrSize)); 
    // gpuErrorcheck(cudaMallocManaged(&nodePointers,sizeof(EDGE_POINTER_TYPE)*vertexArrSize));
    // gpuErrorcheck(cudaMemAdvise(nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemAdviseSetAccessedBy,0));
    infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    gpuErrorcheck(cudaMallocHost(&edgeArray, sizeof(uint)*edgeArrSize));
    // gpuErrorcheck(cudaMallocManaged(&edgeArray,sizeof(uint)*edgeArrSize));
    // gpuErrorcheck(cudaMemAdvise(edgeArray,edgeArrSize*sizeof(uint),cudaMemAdviseSetAccessedBy,0));
    infile.read((char *) edgeArray, sizeof(uint) * edgeArrSize);
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;

    uint* degree;
    bool* isInStatic;
    uint* overloadNodeList;
    bool* isActive;
    uint* value;
    uint* staticnodepointers;
    cout << "initGraphHost()" << endl;
    degree = new uint[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new uint[vertexArrSize];
    isActive = new bool[vertexArrSize];
    value = new uint[vertexArrSize];
    
    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    StaticInfo = getMaxPartionSize(11,edgeArrSize,vertexArrSize,nodePointers,degree,isInStatic);
    for(uint i=0;i<vertexArrSize;i++){
        isActive[i] = 1;
        value[i] = i;
    }

    uint max_static_node = StaticInfo.max_node+1;
    uint max_partition_size = StaticInfo.max_partion_size;
    staticnodepointers = new uint[max_static_node];
    for(uint i=0;i<max_static_node;i++){
        staticnodepointers[i] = (uint)nodePointers[i];
    }

    cudaStream_t StreamStatic, StreamDynamic;
    EDGE_POINTER_TYPE* nodePointersD;
    uint* prefixSumTemp;
    uint* staticNodePointersD;
    uint* staticEdgeListD;
    bool* isInStaticD;
    uint* overloadNodeListD;
    uint* staticNodeListD;
    uint* degreeD;
    bool* isActiveD;
    bool* isStaticActive;
    bool* isOverloadActive;
    uint* valueD;
    cout<<"initGraphDevice()"<<endl;
    cudaMalloc(&prefixSumTemp, vertexArrSize * sizeof(uint));
    gpuErrorcheck(cudaStreamCreate(&StreamStatic));
    gpuErrorcheck(cudaStreamCreate(&StreamDynamic));

    TimeRecord<chrono::milliseconds> preProcess("pre move data");
    preProcess.startRecord();
    gpuErrorcheck(cudaMalloc(&nodePointersD, vertexArrSize*sizeof(EDGE_POINTER_TYPE)));
    gpuErrorcheck(cudaMemcpy(nodePointersD,nodePointers,vertexArrSize*sizeof(EDGE_POINTER_TYPE),cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&staticNodePointersD, max_static_node*sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(staticNodePointersD, staticnodepointers, max_static_node*sizeof(uint),cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(uint)));
    gpuErrorcheck(cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(uint), cudaMemcpyHostToDevice));
    preProcess.endRecord();
    preProcess.print();
    preProcess.clearRecord();
    cudaMalloc(&isInStaticD, vertexArrSize * sizeof(bool));
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&overloadNodeListD, vertexArrSize * sizeof(uint));
    cudaMalloc(&staticNodeListD, vertexArrSize * sizeof(uint));
    cudaMalloc(&degreeD, vertexArrSize * sizeof(uint));
    cudaMalloc(&isActiveD, vertexArrSize * sizeof(bool));
    cudaMalloc(&isStaticActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&isOverloadActive, vertexArrSize * sizeof(bool));
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, isActive, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool));
    cudaMalloc(&valueD, vertexArrSize * sizeof(uint));
    cudaMemcpy(valueD, value, vertexArrSize * sizeof(uint), cudaMemcpyHostToDevice);
    thrust::device_ptr<bool> activeLablingThrust;
    thrust::device_ptr<bool> actStaticLablingThrust;
    thrust::device_ptr<bool> actOverLablingThrust;
    activeLablingThrust = thrust::device_ptr<bool>(isActiveD);
    actStaticLablingThrust = thrust::device_ptr<bool>(isStaticActive);
    actOverLablingThrust = thrust::device_ptr<bool>(isOverloadActive);
    gpuErrorcheck(cudaPeekAtLastError());
    cout << "initGraphDevice() end" << endl;

    cudaDeviceSynchronize();
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    totalProcess.startRecord();
    uint activeNodesNum;
    activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    uint64_t numthreads = 1024;
    long totalduration;
    long overloadduration;
    long staticduration;
    dim3 staticgrid(56,1,1);
    dim3 staticblock(1024,1,1);
    for (int testIndex = 0; testIndex < testTimes; testIndex++){
        //
        cudaDeviceSynchronize();
        cout<<"================="<<"testIndex "<<testIndex<<"================="<<endl;
        uint nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        double overloadsize = 0;
        while(activeNodesNum){
            iter++;
            //cout<<"iter "<<iter<<" activeNodeNum is "<<activeNodesNum<<" ";
            setStaticAndOverloadLabelBool<<<staticgrid,staticblock>>>(vertexArrSize, isActiveD, isStaticActive, isOverloadActive,
                                                        isInStaticD);
            uint staticNodeNum = thrust::reduce(actStaticLablingThrust,
                                                actStaticLablingThrust + vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(prefixSumTemp);
                    
                thrust::exclusive_scan(actStaticLablingThrust, actStaticLablingThrust + vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<staticgrid,staticblock, 0, StreamStatic>>>(vertexArrSize, staticNodeListD, isStaticActive,
                                                                                      prefixSumTemp);
            }
            uint overloadNodeNum = thrust::reduce(actOverLablingThrust,
                                                  actOverLablingThrust + vertexArrSize, 0,
                                                  thrust::plus<uint>());
            //cout<<"staticNodeNum is "<<staticNodeNum<<" overloadNodeNum is "<<overloadNodeNum<<endl;
            if(overloadNodeNum>0){
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(prefixSumTemp);
                thrust::exclusive_scan(actOverLablingThrust, actOverLablingThrust + vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setActiveNodeList<<<staticgrid,staticblock, 0, StreamStatic>>>(vertexArrSize, isOverloadActive, overloadNodeListD,
                                                                               prefixSumTemp);
            }
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<staticgrid, staticblock, 0, StreamStatic>>>(staticNodeNum, staticNodeListD, isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<staticgrid, staticblock, 0, StreamDynamic>>>(overloadNodeNum, overloadNodeListD, isActiveD);
            }
            cudaDeviceSynchronize();
            //cout<<"launch static kernel"<<endl;
            
            staticProcess.startRecord();
            cc_kernelStatic<<<staticgrid, staticblock , 0, StreamStatic>>>(staticNodeNum, staticNodeListD, staticNodePointersD,
                                                                degreeD, staticEdgeListD, valueD, 
                                                                isActiveD, isInStaticD);
            
            if(overloadNodeNum > 0){
                overloadProcess.startRecord();
                // uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                // dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                // uint numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                uint numwarps = (1024*56+32)/32;
                //cout<<"launch overload kernel"<<endl;
                NEW_cc_kernelDynamicSwap_test<<<staticgrid, staticblock ,0,StreamDynamic>>>(overloadNodeNum, overloadNodeListD,
                                                                                        degreeD, valueD, numwarps,
                                                                                        isActiveD,
                                                                                        edgeArray, nodePointersD);

                cudaStreamSynchronize(StreamDynamic);
                overloadProcess.endRecord(); 
                cudaStreamSynchronize(StreamStatic);
                staticProcess.endRecord();
            }
            else{
                cudaDeviceSynchronize();
                staticProcess.endRecord();
                gpuErrorcheck(cudaPeekAtLastError());
            }
            if(staticProcess._isStart()){
                staticProcess.endRecord();
            }
            activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            // uint overloadedges = 0;
            // uint* overloadnodes = new uint[overloadNodeNum];
            // cudaMemcpy(overloadnodes,overloadNodeListD,sizeof(uint)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // for(uint i=0;i<overloadNodeNum;i++){
            //     overloadedges += degree[overloadnodes[i]];
            // }
            // double temp = (double)(overloadedges)*sizeof(uint)/1024;
            // cout<<temp<<endl;
            // overloadsize += temp;
        }
        totalProcess.endRecord();
        totalProcess.print();
        staticProcess.print();
        overloadProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        //cout<< "overloadSize: "<< overloadsize/1024/1024 <<"GB"<<endl;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloadduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        refreshLableAndValue(isActiveD,isStaticActive,isOverloadActive,value,valueD);
        activeNodesNum = thrust::reduce(activeLablingThrust, activeLablingThrust + vertexArrSize, 0,
                                    thrust::plus<uint>());
    }
    cout<<"========TEST OVER========"<<endl;
    cout<<"Test over, average total process time: "<<totalduration/testTimes<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloadduration/testTimes<<"ms"<<endl;
    gpuErrorcheck(cudaPeekAtLastError());
    
    return;
}

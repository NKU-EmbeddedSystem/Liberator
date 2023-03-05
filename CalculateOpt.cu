//
// Created by gxl on 2021/3/24.
//

#include "CalculateOpt.cuh"

template<class T>
__global__ void testRefresh(bool *isActive, T *data, uint size) {
    streamVertices(size, [&](uint id) {
        data[id] = isActive[id];
    });
}

void bfs_opt(string path, uint sourceNode, double adviseRate,int model, int testTimes) {
    cout << "======bfs_opt=======" << endl;
    cout<<"sourceNode: "<<sourceNode<<endl;
    GraphMeta<uint> graph;
    graph.setAlgType(BFS);
    graph.setmodel(model);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    
    graph.setPrestoreRatio(adviseRate, 13);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    cout << "1test!!!!!!!!!" <<endl;
    
    graph.checkNode(sourceNode);
    //getchar();
    // by testing, reduceBool not better than thrust
    //uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);

    cout << "2test!!!!!!!!!" <<endl;
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    //TimeRecord<chrono::milliseconds> MemcpyAsyn("MemcpyAsyn");
    //TimeRecord<chrono::nanoseconds> memtest1("memtest1");
    //TimeRecord<chrono::nanoseconds> memtest2("memtest2");
    
    totalProcess.startRecord();
    cout << "graph.vertexArrSize " << graph.vertexArrSize << endl;
    uint activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    //SIZE_TYPE staticmem = graph.ret_max_partition_size();

    cout << "3test!!!!!!!!!" <<endl;
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();
    cout<<endl;
    EDGE_POINTER_TYPE overloadEdges = 0; 

    cout<<"Start test bfs, total test time: "<<testTimes<<endl;
    uint src=graph.sourceNode;
    long totalduration;
    long overloaduration;
    long staticduration;
    double overloadSize = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        if(src<graph.vertexArrSize)
        graph.setSourceNode(src);
        else
        graph.setSourceNode(sourceNode);
        cout<<"======iter "<<testIndex<<"======"<<endl;
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;

        int iter = 0;
        totalProcess.startRecord();
        while (activeNodesNum) {
            iter++;
            //cout <<"iter "<<iter;
            //cout <<"iter "<<iter<< " activeNodesNum is " << activeNodesNum << " ";
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
                                                                       //cout<<"setStaticAndOverloadLabelBool done!!!!!"<<endl;
            //test static area
            
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
                                                
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                    
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
            }
            //cout<<"staticNodeNum is "<<staticNodeNum;
            
            
            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());

            //cout<<" overloadNodeNum is "<<overloadNodeNum<<endl;
            // if(iter==1){
            //     uint* temp_static = new uint[staticNodeNum];
            //     cudaMemcpy(temp_static,graph.staticNodeListD,staticNodeNum*sizeof(uint),cudaMemcpyDeviceToHost);
            //     for(uint i=0;i<staticNodeNum;i++){
            //         cout<<temp_static[i]<<" ";
            //     }
            //     cout<<endl;  
            // }
                                                
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                
                overloadEdges += overloadEdgeNum;
               // cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }
            // double temp = (double)overloadEdgeNum*sizeof(uint)/1024/1024;
            // cout<<temp<<endl;
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            preProcess.endRecord();
            
            staticProcess.startRecord();

            bfs_kernelStatic<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                graph.staticNodePointerD, graph.degreeD,
                                                                                graph.staticEdgeListD, graph.valueD,
                                                                                graph.isActiveD, graph.isInStaticD);
            
            
            gpuErrorcheck(cudaPeekAtLastError());
            
            if (overloadNodeNum > 0) {
                //MemcpyAsyn.startRecord();
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                cudaDeviceSynchronize();

                gpuErrorcheck(cudaPeekAtLastError());
                staticProcess.endRecord();
                
                overloadProcess.startRecord();
                
                if(graph.model==OLD_MODEL){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    // << i << " graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex] " << graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex] << endl;
                    //cout << i << " graph.partEdgeListInfoArr[i].partEdgeNums " << graph.partEdgeListInfoArr[i].partEdgeNums << endl;
                        overloadMoveProcess.startRecord();
                        gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice));
                        overloadMoveProcess.endRecord();
                        bfs_kernelDynamicPart<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                        graph.partEdgeListInfoArr[i].partStartIndex,
                        graph.partEdgeListInfoArr[i].partActiveNodeNums,
                        graph.overloadNodeListD, graph.degreeD,
                        graph.valueD, graph.isActiveD,
                        graph.overloadEdgeListD,
                        graph.activeOverloadNodePointersD);
                        cudaDeviceSynchronize();
                    }
                }
                else if(graph.model == NEW_MODEL1){
                    
                        for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                            bfs_kernelDynamicPart<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, graph.degreeD,
                            graph.valueD, graph.isActiveD,
                            graph.overloadEdgeList+ graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                            graph.activeOverloadNodePointersD);
                            cudaDeviceSynchronize();
                        }
                        
                }
                else if(model==NEW_MODEL2){
                    uint64_t numthreads = 1024;
                    uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                    uint numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    //cout<<"numwarps of iter"<<iter<<" is "<<numwarps<<endl;
                    bfs_kernelDynamicPart_test2<<<blockDim, numthreads , 0, graph.streamDynamic>>>(
                                overloadNodeNum,numwarps,
                                graph.overloadNodeListD,
                                graph.valueD, graph.degreeD, graph.isActiveD,
                                graph.overloadEdgeList,
                                graph.activeOverloadNodePointersD
                            );
                } else if(model==NEW_MODEL3){
                     uint64_t numthreads = 1024;
                    uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                    uint numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    //cout<<"numwarps of iter"<<iter<<" is "<<numwarps<<endl;
                    bfs_kernelDynamicPart_test<<<blockDim, numthreads , 0, graph.streamDynamic>>>(
                                overloadNodeNum,numwarps,
                                graph.overloadNodeListD,
                                graph.valueD, graph.degreeD, graph.isActiveD,
                                graph.overloadEdgeList,
                                graph.activeOverloadNodePointersD
                            );
                }
                    cudaDeviceSynchronize();
                overloadProcess.endRecord();  
            } else {
                cudaDeviceSynchronize();
                staticProcess.endRecord();
            }
            //cudaDeviceSynchronize();
            preProcess.startRecord();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            //break;
            
        }
        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        double temp = (double)overloadEdges * sizeof(uint)/1024/1024/1024;
        cout << "move overload size : " << temp<<" GB" << endl;
        overloadSize += temp;
        src+=graph.vertexArrSize/testTimes;
        totalduration+=totalProcess.getDuration();
        staticduration+=staticProcess.getDuration();
        overloaduration+=overloadProcess.getDuration();
        totalProcess.clearRecord();
        preProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        forULLProcess.clearRecord();
        overloadMoveProcess.clearRecord();
        overloadEdges=0;
    }
    gpuErrorcheck(cudaPeekAtLastError());
    cout<<"========TEST OVER========"<<endl;
    cout<<"Test over, average total process time: "<<totalduration/testTimes<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloaduration/testTimes<<"ms"<<endl;
    cout<<"average overloadsize: "<<overloadSize/testTimes<<endl;
}

void cc_opt(string path, double adviseRate,int model,int _testTimes) {
    cout << "==========cc_opt==========" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(CC);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 13);
    graph.setmodel(model);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();


    // by testing, reduceBool not better than thrust
    //uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> fillProcess("fillProcess");
    //TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    clock_t start,end; double endtime;
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    totalProcess.startRecord();
    uint activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    cout<<endl;
    totalProcess.clearRecord();
    int testTimes = _testTimes;
    EDGE_POINTER_TYPE overloadEdges = 0;
    long totalduration;
    double overloadduration;
    //long overloaduration;
    long staticduration;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        cout<<"==============="<<"iter "<<testIndex<<"==============="<<endl;
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;
        int iter = 0;

        totalProcess.startRecord();
        while (activeNodesNum) {
            iter++;
            //cout<<"iter "<<iter<<" activeNodeNum is "<<activeNodesNum<<" ";
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << endl;
            }
            //cout << "staticNodeNum is " << staticNodeNum <<" ";
            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            //cout<<"overlaodNodeNum is "<<overloadNodeNum<<endl;
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }

                overloadEdges += overloadEdgeNum;
                //cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }

            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            fillProcess.startRecord();
            cc_kernelStaticSwap<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                   graph.staticNodePointerD,
                                                                                   graph.degreeD,
                                                                                   graph.staticEdgeListD, graph.valueD,
                                                                                   graph.isActiveD, graph.isInStaticD);
            //cudaDeviceSynchronize();

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                cudaDeviceSynchronize();
                fillProcess.endRecord();

                start = clock();
                //overloadProcess.startRecord();
                if(model == OLD_MODEL){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                        overloadMoveProcess.startRecord();
                        gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                        graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                                graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                                cudaMemcpyHostToDevice))
                        overloadMoveProcess.endRecord();
                        cc_kernelDynamicSwap<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                                graph.partEdgeListInfoArr[i].partStartIndex,
                                graph.partEdgeListInfoArr[i].partActiveNodeNums,
                                graph.overloadNodeListD, graph.degreeD,
                                graph.valueD, graph.isActiveD,
                                graph.overloadEdgeListD,
                                graph.activeOverloadNodePointersD);
                        cudaDeviceSynchronize();
                    }
                }
                else if(model == NEW_MODEL1){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++){
                        cc_kernelDynamicSwap<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                                graph.partEdgeListInfoArr[i].partStartIndex,
                                graph.partEdgeListInfoArr[i].partActiveNodeNums,
                                graph.overloadNodeListD, graph.degreeD,
                                graph.valueD, graph.isActiveD,
                                graph.overloadEdgeList,
                                graph.activeOverloadNodePointersD);
                        cudaDeviceSynchronize();
                    }
                }
                else if(model == NEW_MODEL2){
                    uint64_t numthreads = 1024;
                    uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                    uint numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    cc_kernelDynamicSwap_test<<<blockDim,numthreads,0,graph.streamDynamic>>>(
                        overloadNodeNum,graph.overloadNodeListD, graph.degreeD,
                        graph.valueD,numwarps,
                        graph.isActiveD,
                        graph.overloadEdgeList,
                        graph.activeOverloadNodePointersD
                    );
                }

            } else {
                cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            end = clock();
            endtime=(double)(end-start)/CLOCKS_PER_SEC;
	        
            //overloadProcess.endRecord();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            double temp = (double)overloadEdgeNum*sizeof(uint)/1024/1024;
            cout<< temp <<endl;
        }
        totalProcess.endRecord();
        totalProcess.print();
        forULLProcess.print();
        fillProcess.print();
        cout<<"overloadProcess time is "<<endtime*1000<<"ms ";
        //overloadProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        double temp = (double)overloadEdges*sizeof(uint)/1024/1024/1024;
        cout << "move overload size : " << temp << "GB" << endl;
        totalduration+=totalProcess.getDuration();
        staticduration+=fillProcess.getDuration();
        //overloadduration+=overloadProcess.getDuration();
        overloadduration+=endtime;
        totalProcess.clearRecord();
        overloadProcess.clearRecord();
        overloadMoveProcess.clearRecord();
        forULLProcess.clearRecord();
        fillProcess.clearRecord();
        overloadEdges = 0;
    }
    cout<<"========TEST OVER========"<<endl;
    cout<<"Test over, average total process time: "<<totalduration/testTimes<<"ms"<<endl;
    cout<<"average static process time: "<<staticduration/testTimes<<"ms"<<endl;
    cout<<"average overload process time: "<<overloadduration*1000/testTimes<<"ms"<<endl;
    gpuErrorcheck(cudaPeekAtLastError());
}

void sssp_opt(string path, uint sourceNode, double adviseRate,int model,int testTimes) {
    cout << "========sssp_opt==========" << endl;
    GraphMeta<EdgeWithWeight> graph;
    graph.setAlgType(SSSP);
    graph.setmodel(model);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 15);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    gpuErrorcheck(cudaPeekAtLastError());
    graph.checkNodeforSSSP(sourceNode);
    
    // by testing, reduceBool not better than thrust
    //uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");
    totalProcess.startRecord();
    uint activeNodesNum;
    activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    EDGE_POINTER_TYPE overloadEdges = 0;
    uint src = graph.sourceNode;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        cout<<"================="<<"testIndex "<<testIndex<<"================="<<endl;
        cout<<"source: "<<src<<endl;
        graph.setSourceNode(src);
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                        thrust::plus<uint>());
        uint nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        
        while (activeNodesNum) {
            iter++;
            cout<<"iter "<<iter<<" activeNodeNum: "<<activeNodesNum<<endl;
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum;
            }
            //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum;
            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);

                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                overloadEdges += overloadEdgeNum;
                //cout << "iter " << iter << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }
            //double temp = (double)overloadEdgeNum*sizeof(uint)/1024/1024;
            //cout<<temp<<endl;
            if (staticNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum,
                                                                                      graph.staticNodeListD,
                                                                                      graph.isActiveD);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt<<<graph.grid, graph.block, 0, graph.steamStatic>>>(overloadNodeNum,
                                                                                      graph.overloadNodeListD,
                                                                                      graph.isActiveD);
            }

            sssp_kernel<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                           graph.staticNodePointerD, graph.degreeD,
                                                                           graph.staticEdgeListD, graph.valueD,
                                                                           graph.isActiveD);
            //cudaDeviceSynchronize();
            //printf(" overloadnum %ld\n", overloadNodeNum);
            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                if(model==OLD_MODEL){
                    graph.fillEdgeArrByMultiThread(overloadNodeNum);
                    graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);
                }
                cudaDeviceSynchronize();

                if(model==OLD_MODEL){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                        //cout << "graph.partEdgeListInfoArr[i].partEdgeNums " << graph.partEdgeListInfoArr[i].partEdgeNums << endl;
                        overloadMoveProcess.startRecord();
                        gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                        graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                                graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(EdgeWithWeight),
                                                cudaMemcpyHostToDevice))
                        overloadMoveProcess.endRecord();
                        sssp_kernelDynamic<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                                graph.partEdgeListInfoArr[i].partStartIndex,
                                graph.partEdgeListInfoArr[i].partActiveNodeNums,
                                graph.overloadNodeListD, graph.degreeD,
                                graph.valueD, graph.isActiveD,
                                graph.overloadEdgeListD,
                                graph.activeOverloadNodePointersD);
                        cudaDeviceSynchronize();
                    }
                }
                else if(model==NEW_MODEL1){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                        //cout << "graph.partEdgeListInfoArr[i].partEdgeNums " << graph.partEdgeListInfoArr[i].partEdgeNums << endl;
                        sssp_kernelDynamic<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                                graph.partEdgeListInfoArr[i].partStartIndex,
                                graph.partEdgeListInfoArr[i].partActiveNodeNums,
                                graph.overloadNodeListD, graph.degreeD,
                                graph.valueD, graph.isActiveD,
                                graph.edgeArray,
                                graph.activeOverloadNodePointersD);
                        cudaDeviceSynchronize();
                    }
                }
                else if(model==NEW_MODEL2){
                    uint64_t numthreads = 1024;
                    uint64_t numblocks = ((overloadNodeNum * WARP_SIZE + numthreads) / numthreads);
                    dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
                    uint numwarps = blockDim.x*blockDim.y*numthreads / WARP_SIZE;
                    printf("iter %d numblocks %ld numwarps %ld overloadnum %ld\n",iter,numblocks, numwarps, overloadNodeNum);
                    sssp_kernelDynamic_test<<<blockDim, numthreads, 0, graph.streamDynamic>>>(
                        overloadNodeNum, graph.overloadNodeListD, graph.degreeD, numwarps,
                        graph.valueD, graph.isActiveD,
                        graph.edgeArray,
                        graph.activeOverloadNodePointersD
                        );
                    gpuErrorcheck(cudaPeekAtLastError());
                    cudaDeviceSynchronize();
                }

            } else {
                cudaDeviceSynchronize();
            }
            
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            
            //activeNodesNum = thrust::count(graph.activeLablingThrust,graph.activeLablingThrust+graph.vertexArrSize,1);
            // if(n!=activeNodesNum){
            //     cout<<"activeNodeNum error!!!"<<endl;
            //     getchar();
            // }
            // else{
            //     cout<<"activeNodeNum correct"<<endl;
            // }
            nodeSum += activeNodesNum;
            //cout << "iter " << iter << " activeNodesNum " << activeNodesNum << endl;
            //cout<<endl;
        }
        totalProcess.endRecord();
        totalProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        cout << "move overload size : " << overloadEdges * sizeof(EdgeWithWeight) << endl;
        src+=graph.vertexArrSize/testTimes;
        totalProcess.clearRecord();
        forULLProcess.clearRecord();
        overloadMoveProcess.clearRecord();

    }
    gpuErrorcheck(cudaPeekAtLastError());
    //graph.writevalue("oldsssp.txt");
}

void pr_opt(string path, double adviseRate,int model,int testTimes) {
    cout << "=======pr_opt=======" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(PR);
    graph.setmodel(model);
    graph.readDataFromFile(path, true);
    graph.setPrestoreRatio(adviseRate, 17);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();
    graph.checkNodeforPR(1024);

    // by testing, reduceBool not better than thrust
    //uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");
    long Total;
    long Static;
    long Overload;
    totalProcess.startRecord();

    graph.refreshLabelAndValue();
    
    uint activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                    thrust::plus<uint>());
    totalProcess.endRecord();
    cout << "activeNodesNum " << activeNodesNum << endl;
    totalProcess.print();
    totalProcess.clearRecord();
    //int testTimes = 1;
    EDGE_POINTER_TYPE overloadEdges = 0;
    double overloadSize = 0;
    cout<<"=================PR test start================="<<endl;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        cout<<"====="<<testIndex<<" test====="<<endl;
        overloadEdges = 0;
        graph.refreshLabelAndValue();
        activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize, 0,
                                thrust::plus<uint>());
        gpuErrorcheck(cudaPeekAtLastError());
        uint nodeSum = activeNodesNum;
        int iter = 0;
        double totalSum = thrust::reduce(thrust::device, graph.valuePrD, graph.valuePrD + graph.vertexArrSize) / graph.vertexArrSize;
        cout << "totalSum " << totalSum << endl;
        totalProcess.startRecord();
        gpuErrorcheck(cudaPeekAtLastError());
        //cout << "iter " << iter << " first activeNodesNum is " << activeNodesNum << endl;
        unsigned long long overloadedges = 0;
        while (activeNodesNum > 0 && iter<1000) {
            //break;
            iter++;
            overloadedges = 0;
            //cout<<"iter "<<iter;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = thrust::reduce(graph.actStaticLablingThrust,
                                                graph.actStaticLablingThrust + graph.vertexArrSize, 0,
                                                thrust::plus<uint>());
            //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum;
            if (staticNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actStaticLablingThrust, graph.actStaticLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setStaticActiveNodeArray<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.staticNodeListD,
                                                                      graph.isStaticActive,
                                                                      graph.prefixSumTemp);
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum << " ";
            }

            uint overloadNodeNum = thrust::reduce(graph.actOverLablingThrust,
                                                  graph.actOverLablingThrust + graph.vertexArrSize, 0,
                                                  thrust::plus<uint>());
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            //cout<<" overloadNodeNum is "<<overloadNodeNum<<endl;
            if (overloadNodeNum > 0) {
                thrust::device_ptr<uint> tempTestPrefixThrust = thrust::device_ptr<uint>(graph.prefixSumTemp);
                thrust::exclusive_scan(graph.actOverLablingThrust, graph.actOverLablingThrust + graph.vertexArrSize,
                                       tempTestPrefixThrust, 0, thrust::plus<uint>());
                setOverloadNodePointerSwap<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.overloadNodeListD,
                                                                        graph.activeOverloadDegreeD,
                                                                        graph.isOverloadActive,
                                                                        graph.prefixSumTemp, graph.degreeD);
                if(typeid(EDGE_POINTER_TYPE) == typeid(unsigned long long)) {
                    forULLProcess.startRecord();
                    cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                    cudaMemcpyAsync(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyDeviceToHost, graph.streamDynamic);
                    unsigned long long ankor = 0;
                    for(unsigned i = 0; i < overloadNodeNum; ++i) {
                        overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                        if(i > 0) {
                            ankor += overloadDegree[i - 1];
                        }
                        graph.activeOverloadNodePointers[i] = ankor;
                        if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                            cout << i << " : " << graph.activeOverloadNodePointers[i];
                        }
                    }
                    cudaMemcpyAsync(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                    cudaMemcpyHostToDevice, graph.streamDynamic);
                    forULLProcess.endRecord();
                } else {
                    thrust::device_ptr<EDGE_POINTER_TYPE> tempTestNodePointersThrust = thrust::device_ptr<EDGE_POINTER_TYPE>(graph.activeOverloadNodePointersD);
                    
                    thrust::exclusive_scan(graph.actOverDegreeThrust, graph.actOverDegreeThrust + graph.vertexArrSize,
                                           tempTestNodePointersThrust, 0, thrust::plus<uint>());
                    overloadEdgeNum = thrust::reduce(thrust::device, graph.activeOverloadDegreeD,
                                                     graph.activeOverloadDegreeD + overloadNodeNum, 0);
                }
                overloadEdges += overloadEdgeNum;
                //cout << " overloadNodeNum is " << overloadNodeNum << endl;
                //cout << "iter " << iter << " overloadEdgeNum is " << overloadEdgeNum << endl;
            }
            //cout << " overloadNodeNum: " << overloadNodeNum << endl;
            preProcess.endRecord();
            staticProcess.startRecord();
            prSumKernel_static<<<graph.grid, graph.block, 0, graph.steamStatic>>>(staticNodeNum, graph.staticNodeListD,
                                                                                  graph.staticNodePointerD,
                                                                                  graph.staticEdgeListD, graph.degreeD,
                                                                                  graph.outDegreeD, graph.valuePrD,
                                                                                  graph.sumD);
            //cudaDeviceSynchronize();

            if (overloadNodeNum > 0) {
                cudaMemcpyAsync(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                cudaMemcpyAsync(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,
                                overloadNodeNum * sizeof(EDGE_POINTER_TYPE),
                                cudaMemcpyDeviceToHost, graph.streamDynamic);
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);
                cudaDeviceSynchronize();
                staticProcess.endRecord();
                overloadProcess.startRecord();
                if(model==OLD_MODEL){
                    for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                        overloadMoveProcess.startRecord();
                        gpuErrorcheck(cudaMemcpy(graph.overloadEdgeListD, graph.overloadEdgeList +
                                                                      graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],
                                             graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint),
                                             cudaMemcpyHostToDevice))
                        overloadMoveProcess.endRecord();
                        prSumKernel_dynamic<<<graph.grid, graph.block, 0, graph.streamDynamic>>>(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD,
                            graph.activeOverloadNodePointersD,
                            graph.overloadEdgeListD, graph.degreeD, graph.outDegreeD,
                            graph.valuePrD, graph.sumD);
                        cudaDeviceSynchronize();
                    }
                }
                overloadProcess.endRecord();
            } else {
                staticProcess.endRecord();
                cudaDeviceSynchronize();
            }

            preProcess.startRecord();
            prKernel_Opt<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.valuePrD, graph.sumD, graph.isActiveD);
            cudaDeviceSynchronize();
            activeNodesNum = thrust::reduce(graph.activeLablingThrust, graph.activeLablingThrust + graph.vertexArrSize,
                                            0,
                                            thrust::plus<uint>());
            nodeSum += activeNodesNum;
            preProcess.endRecord();
            //cout << "iter " << iter+1 << " activeNodesNum " << activeNodesNum << endl;
            // uint *overloadnodes = new uint[graph.vertexArrSize];
            // cudaMemcpy(overloadnodes,graph.overloadNodeListD,sizeof(uint)*overloadNodeNum,cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // for(uint i=0;i<overloadNodeNum;i++){
            //     overloadedges += graph.degree[overloadnodes[i]];
            // }
            // double temp = (double)overloadedges*sizeof(uint)/1024/1024/1024;
            // cout<<"transfer "<<temp<<" GB"<<endl;
        }
        cout<<"total iter: "<<iter<<endl;
        totalProcess.endRecord();
        totalProcess.print();
        Total+=totalProcess.getDuration();
        preProcess.print();
        staticProcess.print();
        Static+=staticProcess.getDuration();
        overloadProcess.print();
        Overload+=overloadProcess.getDuration();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << endl;
        // double temp = (double)overloadEdges * sizeof(uint)/1024/1024/1024;
        // cout << "move overload size : " << temp <<" GB" << endl;
        // overloadSize+=temp;
    }
    cout<<"=================PR test end================="<<endl;
    cout<<"Average static time: "<<Static/testTimes<<endl;
    cout<<"Average overload time: "<<Overload/testTimes<<endl;
    cout<<"Average Total time: "<<Total/testTimes<<endl;
    //cout<<"Avsrage transfer data "<<overloadSize/testTimes<<endl;
    gpuErrorcheck(cudaPeekAtLastError());
    cudaMemcpy(graph.valuePr,graph.valuePrD,graph.vertexArrSize*sizeof(double),cudaMemcpyDeviceToHost);
    uint num = 0;
    for(uint i=0;i<graph.vertexArrSize;i++){
        if(graph.valuePr[i]==0.15)
        num++;
    }
    cout<<"0.15num: "<<num<<endl;
    //graph.writevalue("Ascetic_PR_GSH_res.txt");
}
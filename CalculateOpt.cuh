//
// Created by gxl on 2021/3/24.
//

#ifndef PTGRAPH_CALCULATEOPT_CUH
#define PTGRAPH_CALCULATEOPT_CUH
#include "GraphMeta.cuh"

#include "gpu_kernels.cuh"
#include "TimeRecord.cuh"
void bfs_opt(string path, uint sourceNode, double adviseRate,int model, int testTimes);
void cc_opt(string path, double adviseRate,int model,int testTimes);
void sssp_opt(string path, uint sourceNode, double adviseRate,int model,int testTimes);
void pr_opt(string path, double adviseRate,int model,int testTimes);
void newbfs_opt(string path, uint sourceNode, double adviseRate,int model, int testTimes);
void newcc_opt(string path, double adviseRate,int model,int testTimes);
void newsssp_opt(string path, uint sourceNode, double adviseRate,int model,int testTimes);
void newpr_opt(string path, double adviseRate,int model,int testTimes);
#endif //PTGRAPH_CALCULATEOPT_CUH

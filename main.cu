#include "CalculateOpt.cuh"
//#include "main.cuh"
#include "constants.cuh"
#include "ArgumentParser.cuh"
#include "New_CC_opt.cuh"
int main(int argc, char** argv) {
    cudaFree(0);cudaDeviceProp devicePropDefined;
	memset(&devicePropDefined, 0, sizeof(cudaDeviceProp));  //设置devicepropDefined的值
	devicePropDefined.major = 6;
	devicePropDefined.minor = 0;
 
	int devicedChoosed;  //选中的设备ID
	
	cudaError_t cudaError;
	cudaChooseDevice(&devicedChoosed, &devicePropDefined);  //查找符合要求的设备ID
	cout << "满足指定属性要求的设备的编号: " << devicedChoosed << endl;
 
	cudaError = cudaSetDevice(devicedChoosed); //设置选中的设备为下文的运行设备
 
	if (cudaError == cudaSuccess)
		cout << "设备选取成功!" << endl;
	else
		cout << "设备选取失败!" << endl;
    cudaGetDevice(&devicedChoosed);  //获取当前设备ID
	cout << "当前使用设备的编号: " << devicedChoosed << endl;

    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "bfs";
    }
    if (arguments.algo == "bfs") {
        cout << "arguments.algo " << arguments.algo << endl;
    cout<<"arguments.sourceNode "<<arguments.sourceNode<<endl;
        if(arguments.model==7)
        newbfs_opt(arguments.input, arguments.sourceNode, arguments.adviseK, arguments.model, arguments.testTimes);
        else
        bfs_opt(arguments.input, arguments.sourceNode, arguments.adviseK,arguments.model,arguments.testTimes);
    } else if (arguments.algo == "cc") {
        if(arguments.model==7)
        //newcc_opt(arguments.input, arguments.adviseK,arguments.model,arguments.testTimes);
        New_CC_opt(arguments.input,arguments.model,arguments.testTimes);
        else
        cc_opt(arguments.input, arguments.adviseK,arguments.model,arguments.testTimes);
    } else if (arguments.algo == "sssp") {
        if(arguments.model==7)
        newsssp_opt(arguments.input, arguments.sourceNode, arguments.adviseK,arguments.model,arguments.testTimes);
        else
        sssp_opt(arguments.input, arguments.sourceNode, arguments.adviseK,arguments.model,arguments.testTimes);
    } else if (arguments.algo == "pr") {
        if(arguments.model==7)
        newpr_opt(arguments.input, arguments.adviseK,arguments.model,arguments.testTimes);
        else
        pr_opt(arguments.input, arguments.adviseK,arguments.model,arguments.testTimes);
    }
    
    return 0;
}


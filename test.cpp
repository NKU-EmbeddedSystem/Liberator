#include "range.cuh"
#include<list>
#include<iostream>
#include<vector>
#include<fstream>
using namespace util::lang;

// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

const int blockDimx=1024;
const int gridDimx=56;
template<typename T>
step_range<T> grid_stride_range(T begin, T end, int blockIdx, int threadIdx) {
    begin += blockDimx * blockIdx + threadIdx;
    return range(begin, end).step(gridDimx * blockDimx);
}

template<typename Predicate>
void streamVertices(int vertices, Predicate p,int blockIdx, int threadIdx) {
    for (auto i : grid_stride_range(0, vertices, blockIdx, threadIdx)) {
        p(i);
    }
}
class vthread{
public:
    int blockIdx;
    int threadIdx;
    std::vector<int> _list;
    vthread(int a,int b)
    {
        blockIdx=a;threadIdx=b;
    }
    vthread()
    {
        blockIdx=threadIdx=0;
    }
    void set(int a,int b){
        blockIdx=a;threadIdx=b;
    }
    void printlist(std::ofstream& outfile)
    {
        if(_list.empty())
        return;
        outfile<<"thread "<<blockDimx * blockIdx + threadIdx<<std::endl;
        for(int i=0;i<_list.size();i++)
        {
            outfile<<_list[i]<<" ";
            //std::cout<<_list[i]<<" ";
        }
        outfile<<std::endl;
    }
};
void test(int nodenum)
{
    const int b_size =blockDimx;
    const int g_size = gridDimx;
    
    vthread** Threads = new vthread*[g_size];

    for(int m=0;m<g_size;m++)
        Threads[m]=new vthread[b_size];
    
    for(int m=0;m<g_size;m++)
        for(int n=0;n<b_size;n++)
        {
            Threads[m][n].set(m,n); 
        }
    for(int m=0;m<g_size;m++){
        for(int n=0;n<b_size;n++){
            streamVertices(nodenum,[&](uint index){
                Threads[m][n]._list.push_back(index);
            },Threads[m][n].blockIdx,Threads[m][n].threadIdx);
        }
    }

    using namespace std;
        std::ofstream outfile("test.txt",ios::out);
        for(int m=0;m<g_size;m++)
            for(int n=0;n<b_size;n++)
            {
                Threads[m][n].printlist(outfile);
            }
        outfile.close();
    
}

int main()
{
    test(blockDimx*gridDimx*2);

    return 0;
}
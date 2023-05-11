## Liberator: A Data Reuse Framework for Out-of-Memory Graph Computing on GPUs
Liberator is an out-of-GPU-memory graph processing framework.

#### Compilation
To compile the Liberator. You need have cmake, g++ and CUDA 11.4 toolkit. 
You should enter the project root dir. 
Create a directory ie. cmake-build-debug. 
Enter it and cmake .. and make to complile the project.
You might need to change the cuda path according to your environment in CMakeList.txt

#### Input graph formats
Liberator accepts the binary CSR format just like :
```
0 4 7 9
1 2 3 4 2 3 4 3 4 5 6
```
#### Data format converter
There is a converter which can convert txt to CSR in the folder converter.
.bcsr is for bfs and cc, .bcsc is for pr(pagerank) and .bwcsr is for sssp.

#### Running
```
$ ./Liberator 
--input datapath 
--type bfs (graph processing algorithm)
--sourceNode 0 (only for bfs and sssp)
--model 7 or 0 (7 is Liberator and the 0 model is our previous work Ascetic )
--tetsTime n (the algorithm will excute n times)
```

#### Publication
[IEEE Transactions on Parallel and Distributed Systems 24 April 2023 doi: DOI 10.1109/TPDS.2023.3268662] Shiyang Li, Ruiqi Tang, Jingyu Zhu, Ziyi Zhao, Xiaoli Gong, Jin Zhang, Wen-wen Wang, and Pen-Chung Yew. Liberator: A Data Reuse Framework for Out-of-Memory Graph Computing on GPUs. 





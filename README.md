## Liberator
Liberator is an out-of-GPU-memory graph processing framework.

#### Compilation
To compile the Liberator. You need have cmake, g++ and CUDA 11.0 toolkit. 
You should enter the project root dir. 
Create a directory ie. cmake-build-debug. 
Enter it and cmake .. and make to complile the project.

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
$ ./ptgraph 
--input datapath 
--type bfs (graph processing algorithm)
--sourceNode 0 (only for bfs and sssp)
--model 7 or 0 (7 is Liberator and the 0 model is our previous work Ascetic )
--tetsTime n (the algorithm will excute n times)
```

#### Publication
[ICPP'21] Ruiqi Tang, Ziyi Zhao, Kailun Wang, Xiaoli Gong, Jin Zhang, Wen-wen Wang, and Pen-Chung Yew. Ascetic: Enhancing Cross-Iterations Data Efficiency in Out-of-Memory Graph Processing on GPUs.





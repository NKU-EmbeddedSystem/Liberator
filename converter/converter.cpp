#include "./globals.hpp"
#include <cstdlib>
#include <ctime>

bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false; 
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}


void convertTxtToByte(string input) {
	cout<<"convertTxtToByte"<<endl;
	clock_t startTime, endTime;
	startTime = clock();
	ifstream infile;
    infile.open(input);
    std::ofstream outfile(input.substr(0, input.length()-3)+"toB", std::ofstream::binary);

	stringstream ss;
    uint max = 0;
    string line;
    ull edgeCounter = 0;
    vector<Edge> edges;
    Edge newEdge;
    ull testIndex = 0;
	uint min = 999999948;
    while(getline( infile, line ))
    {
    	testIndex++;
        ss.str("");
        ss.clear();
        ss << line;
        ss >> newEdge.source;
        ss >> newEdge.end;
		
        //cout << newEdge.source << "  " << newEdge.end << endl;
        edges.push_back(newEdge);
        edgeCounter++;
        if(max < newEdge.source)
            max = newEdge.source;
		if(min > newEdge.source)
			min = newEdge.source;
		
    	if (testIndex % 1000000000 == 1&&testIndex!=1){
    		int billionLines = testIndex / 1000000000;
    		cout << billionLines << " billion lines " << endl;
			vector<Edge>::reverse_iterator it = edges.rbegin();
			cout<<(*it).source<<" "<<(*it).end<<endl;
    		if (billionLines % 2 == 1){
    			outfile.write((char*)edges.data(), sizeof(Edge) * edges.size());
				cout << "clear edges = " << edges.size() << endl;
    			edges.clear();
    		}
    	}

    }
	cout << "max node " << max << endl;
		cout<<"min node "<<min<<endl;
	if (edges.size() > 0){
		cout<<"some rest data "<<edges.size()<<endl;
		vector<Edge>::reverse_iterator it = edges.rbegin();
		cout<<(*it).source<<" "<<(*it).end<<endl;
		outfile.write((char*)edges.data(), sizeof(Edge) * edges.size());
		edges.clear();
	}
		
	cout << "write " << testIndex << " lines" << endl;
	
    outfile.close();
    infile.close();
    endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBCSR(string input) {
	cout << "convertByteToBCSR" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	ull vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	OutEdge* edges = new OutEdge[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++)
	 	degree[i] = 0;

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxsrcNode = 0;
	uint maxNode=0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxsrcNode < byteEdges[i].source){
				maxsrcNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end){
				maxNode = byteEdges[i].end;
			}
			if(maxNode < byteEdges[i].source){
				maxNode = byteEdges[i].source;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	cout << "max source node is "<<maxsrcNode<<endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(ull i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=totalSize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += degree[i];
		if(tempPointer==totalSize)
		tempPointer--;
	}
	for(ull i=0;i<vertexSize;i++)
	{
		if(nodePointers[i]>=totalSize)
		{
			cout<<"pointer error at "<<i<<" with "<<nodePointers[i]<<endl;
			getchar();
		}
	}
	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<totalSize;i++)
	{
		if(edges[i].end>maxNode)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	// outfile.write((char*)&maxsrcNode,sizeof(ull));
	// outfile.write((char*)&maxNode,sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*totalSize);
	outfile.close();
	//cout << "ull size is " << sizeof(ull) << endl;
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}

void convertByteToBCSC(string input) {
	cout << "convertByteToBCSC" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	uint vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	uint* degree = new uint[vertexSize];
	uint *inDegree = new uint[vertexSize];
	OutEdge* edges = new OutEdge[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++) {
	 	degree[i] = 0;
	 	inDegree[i] = 0;
	}

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;
	ull inEdgesize=0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			inDegree[byteEdges[i].end]++;
			inEdgesize++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=inEdgesize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += inDegree[i];
		if(tempPointer==inEdgesize)
		tempPointer--;
	}
	cout << "finish calculate nodePointers " << endl;

	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].end] + nodePointersAnkor[byteEdges[i].end];
	 		edges[location].end = byteEdges[i].source;
	 		nodePointersAnkor[byteEdges[i].end]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<inEdgesize;i++)
	{
		if(edges[i].end>=inEdgesize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}

	std::ofstream outfile(input.substr(0, input.length()-3)+"bcsc", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)degree, sizeof(uint)*vertexSize);
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdge)*inEdgesize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}
void convertByteTollBCSC(string input) {
	cout << "convertByteTollBCSC" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	uint vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	uint* degree = new uint[vertexSize];
	uint *inDegree = new uint[vertexSize];
	llOutEdge* edges = new llOutEdge[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++) {
	 	degree[i] = 0;
	 	inDegree[i] = 0;
	}

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;
	ull inEdgesize=0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			inDegree[byteEdges[i].end]++;
			inEdgesize++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(uint i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=inEdgesize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += inDegree[i];
		if(tempPointer==inEdgesize)
		tempPointer--;
	}
	cout << "finish calculate nodePointers " << endl;

	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].end] + nodePointersAnkor[byteEdges[i].end];
	 		edges[location].end = (ull)(byteEdges[i].source);
	 		nodePointersAnkor[byteEdges[i].end]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<inEdgesize;i++)
	{
		if(edges[i].end>=inEdgesize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}

	std::ofstream outfile(input.substr(0, input.length()-3)+"llbcsc", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)degree, sizeof(uint)*vertexSize);
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(llOutEdge)*inEdgesize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}
void convertByteToBWCSR(string input) {
	cout << "convertByteToBWCSR" << endl;
	srand(0);
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	ull vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	OutEdgeWeighted* edges = new OutEdgeWeighted[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(ull i = 0; i < vertexSize; i++)
	 	degree[i] = 0;

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxNode = 0;

	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxNode < byteEdges[i].source)
			{
				maxNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end)
			{
				maxNode = byteEdges[i].end;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(ull i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=totalSize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += degree[i];
		if(tempPointer==totalSize)
		tempPointer--;
	}

	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * partSize);
		for (uint i = 0; i < partSize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location].end = byteEdges[i].end;
	 		nodePointersAnkor[byteEdges[i].source]++;
	 		edges[location].w8 = rand() % 20;
		}
	}
	for(ull i=0;i<totalSize;i++)
	{
		if(edges[i].end>=vertexSize)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i].end<<endl;
			getchar();
		}
	}
	//delete [] nodePointersAnkor;
	cout << "degree[0] " << degree[0] << endl;
	cout << "degree[1] " << degree[1] << endl;
	//cout << "degree[50000000] " << degree[50000000] << endl;
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"bwcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(OutEdgeWeighted)*totalSize);
	outfile.close();
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}
void convertByteTollBCSR(string input) {
	cout << "convertByteTollBCSR" << endl;
	clock_t startTime, endTime;
	startTime = clock();
	ull totalSize = 10000000000;
	uint partSize = 1000000000;
	ull vertexSize = 1000000000;
	ull* nodePointers = new ull[vertexSize];
	ull* nodePointersAnkor = new ull[vertexSize];
	ull* degree = new ull[vertexSize];
	ull* edges = new ull[totalSize];
	Edge *byteEdges = new Edge[partSize];
	ull counter = 0;
	for(uint i = 0; i < vertexSize; i++)
	 	degree[i] = 0;

	cout << "calculate degree " << endl;
	ifstream infile(input, ios::in | ios::binary);
	infile.seekg(0, std::ios::end);
	ull size = infile.tellg();
	totalSize = size / sizeof(Edge);
	cout << "total edge Size " << totalSize << endl;
	infile.clear();
	infile.seekg(0);
	uint maxsrcNode = 0;
	uint maxNode=0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			degree[byteEdges[i].source]++;
			if (maxsrcNode < byteEdges[i].source){
				maxsrcNode = byteEdges[i].source;
			}
			if (maxNode < byteEdges[i].end){
				maxNode = byteEdges[i].end;
			}
			if(maxNode < byteEdges[i].source){
				maxNode = byteEdges[i].source;
			}
		}
	}
	cout << "max node is " << maxNode << endl;
	cout << "max source node is "<<maxsrcNode<<endl;
	vertexSize = maxNode + 1;
	ull tempPointer = 0;
	for(ull i=0; i<vertexSize; i++)
	{
		nodePointers[i] = tempPointer;
		if(nodePointers[i]>=totalSize)
		{
			cout<<"ndoe error at "<<i<<" nodePointers: "<<nodePointers[i]<<endl;
			getchar();
		}
		tempPointer += degree[i];
		if(tempPointer==totalSize)
		tempPointer--;
	}
	for(ull i=0;i<vertexSize;i++)
	{
		if(nodePointers[i]>=totalSize)
		{
			cout<<"pointer error at "<<i<<" with "<<nodePointers[i]<<endl;
			getchar();
		}
	}
	infile.clear();
	infile.seekg(0);

	cout << "calculate edges " << endl;
	for (uint i = 0; i < vertexSize; ++i)
	{
		nodePointersAnkor[i] = 0;
	}
	counter = 0;
	while(counter < totalSize) {
		cout << counter << " to " << counter + partSize << endl;
		counter += partSize;
		ull readsize;
		if(counter<totalSize)
		readsize=partSize;
		else
		readsize=partSize+totalSize-counter;
		cout<<"readsize: "<<readsize<<endl;
		infile.read((char *) byteEdges, sizeof(Edge) * readsize);
		for (uint i = 0; i < readsize; ++i)
		{
			ull location = nodePointers[byteEdges[i].source] + nodePointersAnkor[byteEdges[i].source];
	 		edges[location] = (ull)(byteEdges[i].end);
	 		nodePointersAnkor[byteEdges[i].source]++;
		}
	}
	//delete [] nodePointersAnkor;
	for(ull i=0;i<totalSize;i++)
	{
		if(edges[i]>maxNode)
		{
			cout<<"edge error at "<<i<<" with "<<edges[i]<<endl;
			getchar();
		}
	}
	
	std::ofstream outfile(input.substr(0, input.length()-3)+"llbcsr", std::ofstream::binary);
	outfile.write((char*)&vertexSize, sizeof(ull));
	outfile.write((char*)&totalSize, sizeof(ull));
	// outfile.write((char*)&maxsrcNode,sizeof(ull));
	// outfile.write((char*)&maxNode,sizeof(ull));
	outfile.write ((char*)nodePointers, sizeof(ull)*vertexSize);
	outfile.write ((char*)edges, sizeof(ull)*totalSize);
	outfile.close();
	//cout << "ull size is " << sizeof(ull) << endl;
	endTime = clock();
    cout << setprecision(9)  << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s | " << "S0:Prepare [edges] OK" << endl;
}
void convertToBEL(string input){
	cout<<"convert to BEL"<<endl;
	string fileName = "/home/lsy/data/"+input+"/"+input+".bwcsr";
	cout<<fileName<<endl;
	ull vertexArrSize;
	ull edgeArrSize;
	cout<<"reading...."<<endl;
	ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &vertexArrSize, sizeof(ull));
    infile.read((char *) &edgeArrSize, sizeof(ull));
    cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << endl;
	ull* nodePointers = new ull[vertexArrSize];
	OutEdgeWeighted* edges =new OutEdgeWeighted[edgeArrSize];
	infile.read((char *) nodePointers, sizeof(ull) * vertexArrSize);
	infile.read((char *) edges, sizeof(OutEdgeWeighted) * edgeArrSize);
	cout<<"copying...."<<endl;
	ull *beldst = new ull[edgeArrSize];
	uint* belval = new uint[edgeArrSize];
	
	for(ull i =0;i<edgeArrSize;i++){
		beldst[i] = (ull)edges[i].end;
		belval[i] = edges[i].w8;
	}
	cout<<"writing bel..."<<endl;
	std::ofstream coloutfile(input+".bel.col", std::ofstream::binary);
	std::ofstream dstoutfile(input+".bel.dst", std::ofstream::binary);
	std::ofstream valoutfile(input+".bel.val", std::ofstream::binary);
	ull placeholder = 0;
	coloutfile.write((char*)&vertexArrSize,sizeof(ull));
	coloutfile.write((char*)&placeholder,sizeof(ull));
	coloutfile.write((char*)nodePointers, sizeof(ull)*vertexArrSize);
	coloutfile.close();
	dstoutfile.write((char*)&edgeArrSize,sizeof(ull));
	dstoutfile.write((char*)&placeholder,sizeof(ull));
	dstoutfile.write((char*)beldst, sizeof(ull)*edgeArrSize);
	dstoutfile.close();
	valoutfile.write((char*)&edgeArrSize,sizeof(ull));
	valoutfile.write((char*)&placeholder,sizeof(ull));
	valoutfile.write((char*)belval, sizeof(uint)*edgeArrSize);
	valoutfile.close();
	cout<<"BEL OK"<<endl;
}
int main(int argc, char** argv)
{
	if(argc!= 2)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}

	string input = string(argv[1]);
	//convertTxtToByte(input);
	//convertByteToBCSR(input);
	//convertByteToBCSC(input);
	//convertByteToBWCSR(input); 
	//convertByteTollBCSC(input);
	//convertByteTollBCSR(input);
	convertToBEL(input);
}

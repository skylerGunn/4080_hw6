#include <cuda_runtime.h>
#include<vector>
#include<iostream>
#include<iterator>
#include<map>
#include<fstream>
#include<string>
#include<cstdlib>
#include<chrono>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#define _USE_MATH_DEFINES
#include<math.h>
//#include<unistd.h>
#include<fcntl.h>
#include<sys/stat.h>
//#include<sys/ipc.h>
//#include<pthread.h>
#include<thread>
#define R 6371
#define TO_RAD (M_PI / 180.0)
#include "math.h"
#define UNIFIED_MATH_CUDA_H

/*__constant__ double xTry[2204];
__constant__ double yTry[2204];
__constant__ double vTry[2204];*/

__device__ double haver(double x1, double y1, double x2, double y2) //Note: deprecated function
{
	double dx, dy, dz;
	y1 -= y2;
	//y1 *= TO_RAD, x1 *= TO_RAD, x2 *= TO_RAD;
	y1 *= (3.141 / 180.0), x1 *= (3.141 / 180.0), x2 *= (3.141 / 180.0);
	dz = sin(x1) - sin(x2);
	dx = cos(y1) * cos(x1) - cos(x2);
	dy = sin(y1) * cos(x1);
	//return dy;
	//return (sqrt(dx * dx + dy * dy + dz * dz) / 2);
	return (asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * R) * 1000; //return haversine distance in
}
//__global__ void kernelIter(cudaTextureObject_t tex, float* listY, float* shiftX, float* shiftY, float* vList, int* pointDis, double radius, int count, int version) {
__global__ void kernelIter(float* listX, float* listY, float* shiftX, float* shiftY, float* vList, int* pointDis, double radius, int count, int version) { //include various options for threads maybe?
//__global__ void kernelIter(double* listX, double* listY, double* shiftX, double* shiftY, double* vList, int* pointDis, double radius, int count, int version) { //include various options for threads maybe?
//__global__ void kernelIter(double* shiftX, double* shiftY, int* pointDis, double radius, int count, int version) { //include various options for threads maybe?
//try hardcoding 32 threads (1 warp) initially?
	//hardcode 32 for tCount at first
	//other versions: 32 blocks 1 thread, 128 block 8 threads, 8 blocks 128 threads, 64 blocks 16 threads, 16 blocks 64 threads
	int start;
	int end;
	//__shared__ double xTry[2204];
	//__shared__ double yTry[2204];
	//__shared__ double vTry[2204];
	//if (threadIdx.x == 0 && blockIdx.x == 0)
	/*for (int i = 0; i < count; i++) {
		xTry[i] = listX[i];
		//yTry[i] = listY[i];
		//vTry[i] = vList[i];
	}*/
	//xTry[threadIdx.x] = listX[threadIdx.x];
	//__syncthreads();
	//printf("res1: %d res2: %d blockIdx %d threadidx %d \n", res1, res2, blockIdx.x, threadIdx.x);
	//return;
	if (version == 0) {
		start = pointDis[(threadIdx.x * 2)];
		end = pointDis[(threadIdx.x * 2) + 1];
	}
	else if (version == 1) {
		start = pointDis[(blockIdx.x * 2)];
		end = pointDis[(blockIdx.x * 2) + 1];
	}
	else if (version == 2) {
		start = pointDis[(threadIdx.x * 128) + (blockIdx.x)];
		end = pointDis[(threadIdx.x * 128) + (blockIdx.x) + 1];
		//printf("start: %d end %d threadIdx: %d blockIdx: %d \n", start, end, (threadIdx.x * 128) + blockIdx.x, blockIdx.x);
	}
	else if (version == 3) {
		start = pointDis[(threadIdx.x) + (blockIdx.x * 128)];
		end = pointDis[(threadIdx.x) + (blockIdx.x * 128) + 1];
	}
	//printf("start: %d end: %d threadidx: %d \n", start, end, threadIdx.x);
	for (int i = start; i <= end; i++) {
		int k = 0;
		float xTemp = listX[i];
		//float xTemp = tex1Dfetch<float>(tex, i);
		float yTemp = listY[i];
		//double yTemp = yTry[i];
		double tempHav = 0;
		//double weight = 0;
		float weight = 0;
		double weightSum = 0;
		double xSum = 0;
		double ySum = 0;
		double dx, dy, dz;
		float x1, x2, y1, y2;
		#pragma unroll 2000

		for (k = 0; k < count; k++) {
			//tempHav = haver(xTemp, yTemp, listX[k], listY[k]); // make into device
			//haver code try
			//y1 = yTemp - yTry[k];
			y1 = yTemp - listY[k];
			y2 = listY[k];
			//y2 = yTry[k];
			x1 = xTemp;
			x2 = listX[k];
			//x2 = tex1Dfetch<float>(tex,k);
			y1 *= (3.141 / 180.0), x1 *= (3.141 / 180.0), x2 *= (3.141 / 180.0);
			dz = sin(x1) - sin(x2);
			dx = cos(y1) * cos(x1) - cos(x2);
			dy = sin(y1) * cos(x1);
			//return dy;
			//return (sqrt(dx * dx + dy * dy + dz * dz) / 2);
			//return (asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * R) * 1000; //return haversine distance in
			tempHav = (double) (asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * 6371) * 1000;
			if (tempHav < radius) {
				weight = vList[k];
				//weight = vTry[k];
				weightSum += weight;
				xSum += listX[k] * weight;
				//xSum += tex1Dfetch<float>(tex,k) * weight;
				ySum += listY[k] * weight;
				//ySum += yTry[k] * weight;
			}
		}
		double tempSX = xSum / weightSum;
		double tempSY = ySum / weightSum;
		shiftX[i] = tempSX;
		shiftY[i] = tempSY;
	}
}
double haversine(double, double, double, double); //compute the haversine distance of 2 points
void doProc(float*, float*, float*, float*, float*, double, int, int, int);
void removeDups(int*, int*, int*);
using namespace std;
map<tuple<double, double>, double> parseFile(string filename, int startIndex);
int main(int argc, char** argv) {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	char* temp;
	int numProcess;
	int startIndex;
	double thresh; //send
	int radius; //send 
	int count; //send
	//double* xList; //send 
	//double* yList; //send
	//double* vList; //send
	float* xList;
	float* yList;
	float* vList;
	if (argc != 6) {
		cout << "error: incorrect argc. Format is <filename> <num threads> <start point> <thresh> <radius> \n";
		return 0;
	}
	temp = strstr(argv[1], ".csv");
	if (temp == NULL) {
		cout << "not a csv file \n";
		return 0;
	}
	numProcess = atoi(argv[2]);
	if (numProcess <= 0) {
		cout << "num processes must be greater than 0 \n";
		return 0;
	}
	startIndex = atoi(argv[3]);
	if (startIndex < 0 || startIndex > 11) {
		cout << "start index must be between 0 to 11 \n";
		return 0;
	}
	thresh = atof(argv[4]);
	if (thresh <= 0 || thresh >= 1) {
		cout << "threshold must be between 0 to 1 \n";
		return 0;
	}
	radius = atoi(argv[5]);
	if (radius <= 0) {
		cout << "radius must be more than 0 \n";
		return 0;
	}
	map<tuple<double, double>, double> data;
	data = parseFile(argv[1], startIndex);
	vector<double> xTList;
	vector<double> yTList;
	vector<double> vTList;
	count = 0;
	map<tuple<double, double>, double>::iterator map_itr = data.begin();
	for (map_itr; map_itr != data.end(); map_itr++) {
		tuple<double, double> curr = map_itr->first;
		//vector<double> feat = map_itr->second;
		double v = map_itr->second;
		//if (feat[startIndex] > thresh) {
		if (v > thresh) {
			double xxx = get<0>(curr);
			double yyy = get<1>(curr);
			xTList.push_back(xxx);
			yTList.push_back(yyy);
			vTList.push_back(v);
			count++;
		}

	}
	cout << "points " << count << "\n";
	int cc = 0;
	/*xList = (double*)malloc(sizeof(double) * count);
	yList = (double*)malloc(sizeof(double) * count);
	vList = (double*)malloc(sizeof(double) * count);*/
	xList = (float*)malloc(sizeof(double) * count);
	yList = (float*)malloc(sizeof(double) * count);
	vList = (float*)malloc(sizeof(double) * count);
	//cout << "coutn " << count << " size: " << vTList.size() << "\n";
	for (cc = 0; cc < count; cc++) {
		xList[cc] = (float) yTList[cc];
		yList[cc] = (float) xTList[cc];
		vList[cc] = (float) vTList[cc];
	}
	int** pointDis = (int**)malloc(sizeof(int*) * numProcess);
	int i;
	int numPoints = count;
	int overRes = numPoints / numProcess;
	int start = 0;
	int end = overRes;
	int remain = numPoints % numProcess;
	int normSize = end;
	for (i = 0; i < numProcess; i++) {
		pointDis[i] = (int*)malloc(sizeof(int) * 2);
		//pointDis[i][0] = (int) (double(((double)count / (double)numProcess)) * (double) i);
		pointDis[i][0] = start;
		pointDis[i][1] = end - 1;
		//pointDis[i][1] = (int) ((double)pointDis[i][0]+(double)((double)count/(double)numProcess) - 1.00);
		start = end;
		end += normSize;
		if (i == (numProcess - 2)) {
			end += remain;
		}
	}
	//gold standard
	//double* shiftX = (double*)malloc(sizeof(double) * count);
	//double* shiftY = (double*)malloc(sizeof(double) * count);
	float* shiftX = (float*)malloc(sizeof(float) * count);
	float* shiftY = (float*)malloc(sizeof(float) * count);
	int k;
	/*for (i = 0; i < count; i++) {
		//clustList[i] = (int*)calloc(count, sizeof(int));
		shiftX[i] = 0;
		shiftY[i] = 0;
	}*/
	double sumShift = 0;
	int* temper = (int*)malloc(sizeof(int) * numProcess * 2);
	for (int i = 0; i < (2 * numProcess); i++) {
		temper[i] = pointDis[i / 2][0];
		temper[i + 1] = pointDis[i / 2][1];
		//cout << "i " << i << " start " << temper[i] << " end " << temper[i + 1] << "\n";
		i++;
	}
	//return 0;
	int* cudaTemp;
	cudaMalloc(&cudaTemp, sizeof(int) * numProcess * 2);
	//convert to floats
	float* x2;
	float* y2;
	float* shiftX2;
	float* shiftY2;
	float* vList2;
	
	/*double* x2;
	double* y2;
	double* shiftX2;
	double* shiftY2;
	double* vList2;*/
	int* temper2;
	/*cudaMalloc(&x2, sizeof(double) * count);
	cudaMalloc(&y2, sizeof(double) * count);
	cudaMalloc(&shiftX2, sizeof(double) * count);
	cudaMalloc(&shiftY2, sizeof(double) * count);
	cudaMalloc(&vList2, sizeof(double) * count);*/
	cudaMalloc(&x2, sizeof(float) * count);
	cudaMalloc(&y2, sizeof(float) * count);
	cudaMalloc(&shiftX2, sizeof(float) * count);
	cudaMalloc(&shiftY2, sizeof(float) * count);
	cudaMalloc(&vList2, sizeof(float) * count);
	cudaMalloc(&temper2, sizeof(int) * numProcess * 2);
	//try cuda texture with x2
	/*cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = x2;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = sizeof(float) * count;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);*/

	//cudaMemcpy(x2, xList, sizeof(double) * count, cudaMemcpyHostToDevice);
	//cudaMemcpy(y2, yList, sizeof(double) * count, cudaMemcpyHostToDevice);
	//cudaMemcpy(vList2, vList, sizeof(double) * count, cudaMemcpyHostToDevice);

	cudaMemcpy(x2, xList, sizeof(float) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(y2, yList, sizeof(float) * count, cudaMemcpyHostToDevice);
	cudaMemcpy(vList2, vList, sizeof(float) * count, cudaMemcpyHostToDevice);
	/*cudaMemcpyToSymbol(xTry, xList, sizeof(double)* count);
	cudaMemcpyToSymbol(yTry, yList, sizeof(double) * count);
	cudaMemcpyToSymbol(vTry, vList, sizeof(double) * count);*/
	cudaMemcpy(temper2, temper, sizeof(int) * numProcess * 2, cudaMemcpyHostToDevice);
	cudaError_t tempErr;
	sumShift = 0;
	if (numProcess == 32) {
		clock_t timeStart = clock();
		kernelIter << <1, 32 >> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 0);
		cudaMemcpy(shiftX, shiftX2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		cudaMemcpy(shiftY, shiftY2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		for (int i = 0; i < count; i++) {
			sumShift += haversine((double) xList[i], (double) yList[i], (double) shiftX[i], (double) shiftY[i]);
		}
		clock_t stop1 = clock();
		cout << "shift: " << sumShift << " time for 1 block 32 threads " << (double) (stop1 - timeStart)/CLOCKS_PER_SEC << "\n";
		clock_t start2 = clock();
		kernelIter << <32, 1 >> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 1);
		sumShift = 0;
		cudaMemcpy(shiftX, shiftX2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		cudaMemcpy(shiftY, shiftY2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		for (int i = 0; i < count; i++) {
			sumShift += haversine((double)xList[i], (double)yList[i], (double)shiftX[i], (double)shiftY[i]);
		}
		clock_t stop2 = clock();
		cout << "shift: " << sumShift << " time for 32 blocks 1 thread " << (double)(stop2 - start2) / CLOCKS_PER_SEC << "\n";
		sumShift = 0;
		//return 0;
	}
	else if (numProcess == 512) {
		clock_t timeStart = clock();
		kernelIter << <8, 128 >> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 3);
		//kernelIter << <8, 128>> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 3);
		//kernelIter << <8, 128 >> > (shiftX2, shiftY2, temper2, radius, count, 3);
		//tempErr = cudaDeviceSynchronize();
		
		//cout << "err " << cudaGetErrorString(tempErr) << "\n";
		
		//cudaMemcpy(shiftX, shiftX2, sizeof(double) * count, cudaMemcpyDeviceToHost);
		//cudaMemcpy(shiftY, shiftY2, sizeof(double) * count, cudaMemcpyDeviceToHost);

		cudaMemcpy(shiftX, shiftX2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		cudaMemcpy(shiftY, shiftY2, sizeof(float)* count, cudaMemcpyDeviceToHost);
		for (int i = 0; i < count; i++) {
			sumShift += haversine((double) xList[i], (double) yList[i], (double) shiftX[i], (double) shiftY[i]);
			//sumShift += haversine(xList[i], yList[i], shiftX[i], shiftY[i]);
			//cout << "i " << i << " shiftx " << shiftX[i] << " shift y " << shiftY[i] << "\n";
		}
		clock_t stop1 = clock();
		cout << "shift: " << sumShift << " time for 8 block 128 threads " << (double)(stop1 - timeStart) / CLOCKS_PER_SEC << "\n";
		sumShift = 0;
		clock_t start2 = clock();
		//kernelIter << <128, 8>> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 2);
		kernelIter << <128, 8 >> > (x2, y2, shiftX2, shiftY2, vList2, temper2, radius, count, 2);
		//kernelIter << <128, 8 >> > (shiftX2, shiftY2, temper2, radius, count, 2);
		//cudaMemcpy(shiftX, shiftX2, sizeof(double) * count, cudaMemcpyDeviceToHost);
		//cudaMemcpy(shiftY, shiftY2, sizeof(double) * count, cudaMemcpyDeviceToHost);
		cudaMemcpy(shiftX, shiftX2, sizeof(float) * count, cudaMemcpyDeviceToHost);
		cudaMemcpy(shiftY, shiftY2, sizeof(float) * count, cudaMemcpyDeviceToHost);
		for (int i = 0; i < count; i++) {
			//sumShift += haversine(xList[i], yList[i], shiftX[i], shiftY[i]);
			sumShift += haversine((double)xList[i], (double)yList[i], (double)shiftX[i], (double)shiftY[i]);
			//cout << "i " << i << " shiftx " << shiftX[i] << " shift y " << shiftY[i] << "\n";
		}
		clock_t stop2 = clock();
		cout << "shift: " << sumShift << " time for 128 blocks 8 threads " << (double)(stop2 - start2) / CLOCKS_PER_SEC << "\n";
		sumShift = 0;
	}
	//now try other versions:
	//other versions: 32 blocks 1 thread, 128 block 8 threads, 8 blocks 128 threads, 64 blocks 16 threads, 16 blocks 64 threads
	
	//one iteration
	clock_t start2 = clock();
	thread* tList = new thread[numProcess];
	//cout << "num proc : " << numProcess << "\n";
	for (k = 0; k < numProcess; k++) {
		//cout << "making thread " << k << "\n";
		tList[k] = thread(doProc,xList,yList,shiftX,shiftY,vList,radius,count,pointDis[k][0], pointDis[k][1]);
	}
	for (k = 0; k < numProcess; k++) {
		tList[k].join();
	}
	//free(tList);
	cout << "num points: " << count << "\n";
	for (int i = 0; i < count; i++) {
		sumShift += haversine((double)xList[i], (double)yList[i], (double)shiftX[i], (double)shiftY[i]);
	}
	clock_t stop = clock();
	cout << "shift: " << sumShift << " time for cpu " << (double)(stop - start2) / CLOCKS_PER_SEC << "\n";
	sumShift = 0;
	/*for (int i = 0; i < count; i++) {
		if (shiftX[i] != xList[i] || shiftY[i] != yList[i]) {
			cout << " i " << i << " x " << xList[i] << " y " << yList[i] << " x2 " << shiftX[i] << " y2 " << shiftY[i] << "\n";
		}
	}*/
	//now copy stuff
	//make into 1d
	/*for (int i = 0; i < count; i++) {
		if (shiftX[i] != xList[i] || shiftY[i] != yList[i]) {
			cout << " i " << i << " x " << xList[i] << " y " << yList[i] << " x2 " << shiftX[i] << " y2 " << shiftY[i] << "\n";
		}
	}*/
	cudaFree(&x2);
	cudaFree(&y2);
	cudaFree(&shiftX2);
	cudaFree(&shiftY2);
	cudaFree(&vList2);
	cudaFree(&temper2);
	free(shiftX);
	free(shiftY);
	free(vList);
	free(temper);
	for (int i = 0; i < numProcess; i++) {
		free(pointDis[i]);
	}
	free(pointDis);
	return 0;
}
map<tuple<double, double>, double> parseFile(string filename, int startIndex) {
	//map<tuple<double,double>, vector<double>> dataMap;
	map<tuple<double, double>, double> dataMap;
	string line;
	ifstream dataFile(filename);
	if (dataFile.is_open()) {
		while (getline(dataFile, line)) {
			size_t pos = 0;
			pos = line.find(":");
			double x = stod(line.substr(0, pos));
			line.erase(0, pos + 1);
			pos = line.find(":");
			double y = stod(line.substr(0, pos));
			line.erase(0, pos + 1);
			double weight;
			for (int ttt = 0; ttt < startIndex; ttt++) {
				pos = line.find(":");
				double x1 = stod(line.substr(0, pos));
				line.erase(0, pos + 1);
			}
			pos = line.find(":");
			weight = stod(line.substr(0, pos));
			dataMap.insert(pair<tuple<double, double>, double>(tuple<double, double>(x, y), weight));
		}
		dataFile.close();
	}
	return dataMap;
}


double haversine(double x1, double y1, double x2, double y2) {
	double dx, dy, dz;
	y1 -= y2;
	y1 *= TO_RAD, x1 *= TO_RAD, x2 *= TO_RAD;
	dz = sin(x1) - sin(x2);
	dx = cos(y1) * cos(x1) - cos(x2);
	dy = sin(y1) * cos(x1);
	//return dy;
	//return (sqrt(dx * dx + dy * dy + dz * dz) / 2);
	return (asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * R) * 1000; //return haversine distance in m
}


//main idea: from a cluster of points, shift each point to get to center of the cluster using the mean, returns amount point moved
void doProc(float* xList, float* yList, float* xShift, float* yShift, float* vList, double radius, int count, int sPoint, int ePoint) {
//void doProc(double* xList, double* yList, double* xShift, double* yShift, double* vList, double radius, int count, int sPoint, int ePoint) {
	int j = 0;
	for (j = sPoint; j <= ePoint; j++) {
		int i = 0;
		//double xTemp = xList[pointIndex];
		//double yTemp = yList[pointIndex];
		float xTemp = xList[j];
		float yTemp = yList[j];
		double tempHav = 0;
		float weight = 0;
		double weightSum = 0;
		float xSum = 0;
		float ySum = 0;
		for (i = 0; i < count; i++) {
			tempHav = haversine((double) xTemp, (double) yTemp, (double) xList[i], (double) yList[i]);
			//cout << "p " << pointIndex << " i " << i << " x1 " << xTemp << " y1 " << yTemp << " x2 " << xList[i] << " y2 " << yList[i] << " hav " << haversine(xTemp,yTemp,xList[i],yList[i]) << "\n";
			//tempHav = haversine(xTemp, yTemp, xShift[i], yShift[i]);
			if (tempHav < radius) {
				weight = vList[i];
				weightSum += weight;
				xSum += xList[i] * weight;
				ySum += yList[i] * weight;
				//clustList[j][0]++;
				//clustList[j][clustList[j][0] - 1] = i;
			}
			//}
		}
		//printf("weightSum %lf point %d \n", weightSum, pointIndex);
		double tempSX = xSum / weightSum;
		double tempSY = ySum / weightSum;
		//cout << "point: " << pointIndex << " xTot " << xSum << " yTot " << ySum << "\n";
		//if (pointIndex < 50) 
		//cout << "p " << pointIndex << " x1 " << xTemp << " y1 " << yTemp << " x2 " << tempSX << " y2 " << tempSY << " hav " << haversine(xTemp, yTemp, tempSX, tempSY) << "\n";
		//xShift[pointIndex] = tempSX;
		//yShift[pointIndex] = tempSY;
		xShift[j] = tempSX;
		yShift[j] = tempSY;
		//cout << "j " << j << " xShi " << xShift[j] << " ySHi " << yShift[j] << "\n";
		//printf("point: %d xTot: %lf yTot: %lf \n", pointIndex, xSum, ySum);
		//printf("xShift: %lf yShift: %lf xList: %lf yList: %lf point: %d id: %d xSum: %lf ySum: %lf weightSum: %lf \n", xShift[pointIndex], yShift[pointIndex], xList[pointIndex], yList[pointIndex], pointIndex, id, xSum, ySum, weightSum);
	}
}
void removeDups(int* result, int* lis1, int* lis2) {//note: may get off by 1 error/slightly off since does the center of the cluster have a point or not?
	int temp;
	int i, j;
	int size = 1;
	int addFlag = 1;
	int addFlag2 = 1;
	int z;
	for (i = 1; i < lis1[0] - 1; i++) {
		temp = lis1[i];
		if (temp != -1) {
			for (j = 1; j < lis2[0] - 1; j++) {
				if (temp == lis2[j]) {
					addFlag = 0;
					break;
				}
			}
			if (addFlag == 1) {
				for (z = 1; z < size; z++) {
					if (temp == result[z]) {
						//already added
						addFlag2 = 0;
						break;
					}
				}
				if (addFlag2 == 1) {
					//add to list
					result[size] = temp;
					//lis1[i] = -1;
					size++;
				}
			}
		}
		addFlag2 = 1;
		addFlag = 1;
	}
	for (i = 1; i < lis2[0] - 1; i++) {
		temp = lis2[i];
		if (temp != -1) {
			for (j = 1; j < lis1[0] - 1; j++) {
				if (temp == lis1[j]) {
					addFlag = 0;
					break;
				}
			}
			if (addFlag == 1) {
				for (z = 1; z < size; z++) {
					if (temp == result[z]) {
						addFlag2 = 0;
						break;
					}
				}
				if (addFlag2 == 1) {
					result[size] = temp;
					//lis2[i] = -1;
					size++;
				}
			}
		}
		addFlag = 1;
		addFlag2 = 1;
	}
	result[0] = size;
}


#include "cudaLib.cuh"
#include "lab1.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) y[i] =  scale * x[i] + y[i]; 
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";
	// Create float pointer
	float *x;
	float *y;
	float *z;
	float *gpu_x;
	float *gpu_y;
	float scale = (float) (rand() % (1000));
	uint64_t size = vectorSize * sizeof(float);

	x = (float*)malloc(size);
	y = (float*)malloc(size);
	z = (float*)malloc(size);

	for(int z = 0; z < vectorSize; z++){
		x[z] =  (float) (rand() % (1000));
		y[z] =  (float) (rand() % (1000));

	}

	printf("\n Adding vectors : \n");
	printf(" scale = %f\n", scale);
	printf(" x = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", x[i]);
	}
	printf(" ... }\n");
	printf(" y = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", y[i]);
	}
	printf(" ... }\n");

	cudaMalloc((void **) &gpu_x, size);
	cudaMalloc((void **) &gpu_y, size);

	cudaMemcpy(gpu_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = PART1_TPB;
	int blocks = vectorSize;
	blocks = blocks/threadsPerBlock;
	//printf("\nblocks %d", blocks);
	if(vectorSize % threadsPerBlock) {
		blocks++;
	}

	if(blocks == 1){
		//threadsPerBlock = vectorSize;
	}

	//printf("blocks %d, tpb: %d", blocks, threadsPerBlock);

	saxpy_gpu<<<blocks,threadsPerBlock>>>(gpu_x, gpu_y, scale, vectorSize);

	cudaMemcpy(z, gpu_y, size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_x);
	cudaFree(gpu_y);

	printf(" z = { ");
	for (int i = 0; i < 5; ++i) {
		printf("%3.4f, ", z[i]);
	}
	printf(" ... }\n");
	

	int errorCount = verifyVector(x, y, z, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	free(x);
	free(y);
	free(z);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here

	curandState_t rng;
	curand_init(clock64(), threadIdx.x, 0, &rng);

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i: %d", i);
	uint64_t sum = 0;

	if(i < pSumSize){

	for (int z = 0; z < sampleSize; z++){
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		sum += int(x * x + y * y) == 0;
	}

	pSums[i] = sum;

	}

}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	for(int z = 0; z < reduceSize; z++){
			sum += pSums[(i * reduceSize) + z];
		totals[i] = sum;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//printf("\ngenerateThreadCount: %d", generateThreadCount);
	//printf("\nreduceThreadCount: %d", reduceThreadCount);
	//printf("\nsampleSize: %d", sampleSize);
	//printf("\nreduceSize: %d", reduceSize);
	
	
	double approxPi = 0;

	uint64_t numpsums = generateThreadCount;
	uint64_t numpreduce = reduceThreadCount;

	uint64_t sumsize = numpsums * sizeof(uint64_t);
	uint64_t reducesize = numpreduce * sizeof(uint64_t);

	uint64_t *totals;

	uint64_t *psums_gpu;
	uint64_t *totals_gpu;

	totals = (uint64_t*)malloc(reducesize);

	cudaMalloc((void **) &psums_gpu, sumsize);
	cudaMalloc((void **) &totals_gpu, reducesize);

	uint64_t pval = PART1_TPB;
	uint64_t rval = pval / 4;
	

	generatePoints<<<generateThreadCount/pval,pval>>>(psums_gpu, numpsums, sampleSize);
	reduceCounts<<<reduceThreadCount/rval,rval>>>(psums_gpu, totals_gpu, numpsums, reduceSize);

	cudaMemcpy(totals, totals_gpu, reducesize, cudaMemcpyDeviceToHost);

	//printf("\ntotals: %d", totals[0]);

	uint64_t totalhits = 0;
	for (uint64_t z = 0; z < reduceThreadCount; z++) {

		totalhits += totals[z];
		//printf("totals[%d]: %d", z, totals[z]);
		//printf("\ntotalhits: %d", totalhits);
	}

	//printf("\ntotalhits: %u", totalhits);
	//printf("\ngenerateThreadCount: %d", generateThreadCount);
	//printf("\nsampleSize: %d", sampleSize);
	//printf("\ntotalhits / generateThreadCount: %f", (double)totalhits / (double)generateThreadCount);


	approxPi = ((double)totalhits / (double)generateThreadCount / (double)(sampleSize));
	approxPi = approxPi * 4.0f;

	cudaFree(psums_gpu);
	cudaFree(totals_gpu);

	free(totals);

	return approxPi;
}

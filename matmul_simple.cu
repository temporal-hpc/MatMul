#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 32
#define NUM char

__global__ void MatMul(int n, NUM *a, NUM *b, int *c){
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	
	int sum = 0;

	for(int i = 0; i < n; i++){
		sum += a[tidy * n + i] * b[i * n + tidx];
	}
	c[tidy * n + tidx] = sum;
}

__global__ void SharedMatMul(int n, NUM *a, NUM *b, int *c){
	__shared__ NUM as[BLOCKSIZE*BLOCKSIZE*4];
	__shared__ NUM bs[BLOCKSIZE*BLOCKSIZE*4];
	__shared__ int cs[BLOCKSIZE*BLOCKSIZE];
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int lx = threadIdx.x;
	int ly = threadIdx.y;

	// COPY FROM GLOBAL MEMORY
	cs[ly * BLOCKSIZE + lx] = 0;
	for(int i = 0; i < n; i+=BLOCKSIZE){
		as[4*(ly * BLOCKSIZE + lx)] = a[tidy * n + (lx + i)];
		bs[4*(ly * BLOCKSIZE + lx)] = b[(ly + i) * n  + tidx];
		__syncthreads();
		for(int j = 0; j < BLOCKSIZE; j++){
			cs[BLOCKSIZE * ly + lx] += as[4*(BLOCKSIZE * ly + j)] * bs[4*(BLOCKSIZE * j + lx)]; 
		}
		__syncthreads();
		//c[tidy * n + tidx] += cs[BLOCKSIZE * ly + lx];
	}
	//__syncthreads();
	c[tidy * n + tidx] += cs[BLOCKSIZE * ly + lx];
}

__host__ void fillMatrix(NUM *a, int n){
	for(int i = 0; i < n * n; i++){
		a[i] = rand()%3 - 1;
	}
}

__host__ void fillMatrixZeros(int *a, int n){
	for(int i = 0; i < n*n; i++){
		a[i] = 0;
	}
}

__host__ void hostMatMul(int n, NUM *a, NUM *b, int *c){
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			int val = c[i*n + j];
			for(int k=0; k<n; ++k){
				val += a[i*n+k] * b[k*n +j];
			 }
			c[i*n + j] = val;
		}
	}
}

__host__ int verify(int n, int *a, int *b){
	int cont = 0;
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			cont++;
			if(a[i*n+j]!=b[i*n+j]){
				printf("FALLO %d\t%d!=%d\n",cont , a[i*n+j], b[i*n+j]);
				//return 1;
			}	
		}
	}
	return 0;
}

__host__ void printmat(int *a, int n, const char *name){
	printf("mat %s:\n", name);
	for(int i=0; i<n; ++i){
 		for(int j=0; j<n; ++j){
			printf("%i ", a[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
}


int main(int argc, char *argv[]){
	if (argc != 3){
		printf("Ejecute como ./prog N\n");
		return EXIT_FAILURE;
	}	
	int n = atoi(argv[1]);
	srand(atoi(argv[2]));

	NUM *a, *b, *a_d, *b_d;
	int *c, *d, *c_d, *d_d, *matmul_simple, *matmul_shared;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	a = (NUM*)malloc(sizeof(NUM)*n*n);
	b = (NUM*)malloc(sizeof(NUM)*n*n);
	c = (int*)malloc(sizeof(int)*n*n);
	d = (int*)malloc(sizeof(int)*n*n);
	matmul_simple = (int*)malloc(sizeof(int)*n*n);
	matmul_shared = (int*)malloc(sizeof(int)*n*n);

	fillMatrix(a, n);
	fillMatrix(b, n);
	//fillMatrixZeros(c, n);

	cudaMalloc(&a_d, sizeof(NUM)*n*n);
	cudaMalloc(&b_d, sizeof(NUM)*n*n);
	cudaMalloc(&c_d, sizeof(int)*n*n);
	cudaMalloc(&d_d, sizeof(int)*n*n);
	
	cudaMemcpy(a_d, a, sizeof(NUM) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(NUM) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	
	dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 grid(n/BLOCKSIZE, n/BLOCKSIZE, 1);
/*
	printf("block: %d %d %d\n", block.x, block.y, block.z);
	printf("grid : %d %d %d\n", grid.x, grid.y, grid.z);
*/
	
	cudaEventRecord(start);
	MatMul<<<grid,block>>>(n ,a_d, b_d, c_d);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
//	printf("GPU Simple...ok! in %f ms\n", ms);
	printf("%f\n", ms);
/*
	cudaEventRecord(start);
	SharedMatMul<<<grid,block>>>(n ,a_d, b_d, d_d);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Shared Memory...ok! in %f ms\n", ms);

	cudaMemcpy(matmul_shared, d_d, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(matmul_simple, c_d, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

//	hostMatMul(n, a, b, c);

	if(verify(n, matmul_simple, matmul_shared) != 0){
        	fprintf(stderr, "error verifying result\n");
        	exit(EXIT_FAILURE);
    	}
*/
	return 0;
}

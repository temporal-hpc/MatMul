#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void MatMul_TensorCore(int n, signed char *a, signed char *b, int* c){
	int tx = threadIdx.x;
	unsigned int tid =  blockDim.x*threadIdx.y + threadIdx.x;
	int2 wid = {tx>>5, (int)threadIdx.y};
	int offC = (wid.y*16*n + wid.x*16);

	__shared__ signed char mata[4096];
	__shared__ signed char matb[4096];

	wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
	wmma::fill_fragment(c_frag, 0);

	for (int j = 0; j < n; j+=64){
		
		//wmma::load_matrix_sync(a_frag, a + j + wid.x*16 + wid.y*n*16, n);
		//wmma::store_matrix_sync(mata + wid.y*16*64 + wid.x*16, a_frag, 64, wmma::mem_row_major);
		
		//wmma::load_matrix_sync(b_frag, b + wid.x*16 + wid.y*n*16*j, n);
		//wmma::store_matrix_sync(matb + wid.y*16*64 + wid.x*16, b_frag, 64, wmma::mem_row_major);
		for (int p=0; p<8; p++){
			int tsd = tid & 63; // tid % 64
			int tsy = tid >> 6; // tid / 2^6 -> (tid//64)
			mata[tid + p*512] = a[tsd + n*p*8 + tsy*n + j + blockIdx.y*n*64];
			matb[tid + p*512] = b[tsd + n*p*8 + tsy*n + j*n + blockIdx.x*64];
		}

		__syncthreads();

		for(int i = 0; i < 64; i+=16){
			int offA = i + wid.y*16*64; // >> 
			int offB = i*64 + wid.x*16; // vv
			wmma::load_matrix_sync(a_frag, mata + offA, 64);
			wmma::load_matrix_sync(b_frag, matb + offB, 64);
	
			wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);	
		}
		__syncthreads();
	}

	wmma::store_matrix_sync(c + wid.y*16*n + wid.x*16 + blockIdx.x*64 + blockIdx.y*n*64, c_frag, n, wmma::mem_row_major);
	
}


__host__ void fillMatrix(signed char *a, int n){
	for(int i = 0; i < n * n; i++){
		a[i] = rand()%3 - 1;
	}
}

__host__ void fillMatrixZeros(int *a, int n){
	for(int i = 0; i < n*n; i++){
		a[i] = 0;
	}
}

__host__ void hostMatMul(int n, signed char *a, signed char *b, int *c){
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			int val = 0;//c[i*n + j];
			for(int k=0; k<n; ++k){
				val +=(unsigned int)(a[i*n+k] * b[k*n +j]);
			}
			//printf("%i \n", val);
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
			}else{	
				//printf("PASS %d\t%d!=%d\n",cont , a[i*n+j], b[i*n+j]);
				continue;
			}	
		}
	}
	return 0;
}

__host__ void printmat(float *a, int n, const char *name){
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

	signed char *a, *b; 
	signed char *a_d, *b_d;
	int *c, *c_d, *matmul_tc, *matmul_host;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	a = (signed char*)malloc(sizeof(signed char)*n*n);
	b = (signed char*)malloc(sizeof(signed char)*n*n);

	c = (int*)malloc(sizeof(int)*n*n);
	matmul_tc = (int*)malloc(sizeof(int)*n*n);
	matmul_host = (int*)malloc(sizeof(int)*n*n);

	fillMatrix(a, n);
	fillMatrix(b, n);
	fillMatrixZeros(c, n);

	cudaMalloc(&a_d, sizeof(signed char)*n*n);
	cudaMalloc(&b_d, sizeof(signed char)*n*n);
	cudaMalloc(&c_d, sizeof(int)*n*n);
	
	cudaMemcpy(a_d, a, sizeof(signed char) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(signed char) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	
	dim3 block(128, 4, 1); // 128 -> 4   -> 32  -> 16 X 16 
	dim3 grid(n/64, n/64, 1);
/*
	printf("block: %d %d %d\n", block.x, block.y, block.z);
	printf("grid : %d %d %d\n", grid.x, grid.y, grid.z);
*/
	
	cudaEventRecord(start);
	MatMul_TensorCore<<<grid, block>>>(n ,a_d, b_d, c_d);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&ms, start, stop);
//	printf("GPU Tensor Core...ok! in %f ms\n", ms);
	printf("%f\n", ms);

	cudaMemcpy(matmul_tc, c_d, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	
//	hostMatMul(n,a,b,matmul_host);
//	verify(n, matmul_tc, matmul_host);

	return EXIT_SUCCESS;
}

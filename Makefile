all:
	nvcc -o prog -arch=sm_75 -lcublas -lcurand matmul_tensor-core.cu	
sm:
	nvcc -o smprog -arch=sm_75 -lcublas -lcurand matmul_simple.cu	


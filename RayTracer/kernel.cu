#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>
#include <fstream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// 保证在退出前重置CUDA设备
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(float* fb, int max_x, int max_y) {
	// i,j是当前thread所在grid中的唯一标识，以下计算是当grid和block二维时的计算方式
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return; //分配的总线程数超过了1200x800，超过的部分不需要参与计算
	int pixel_index = j * max_x * 3 + i * 3; //pixel_index是把照片上(i,j)像素映射到一维的fb中的下标变量
	// 利用当前线程计算照片中的像素(i,j)，分别计算r,g,b
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
}

int main() {
	// 渲染一张 1200 x 600 个像素的照片
	int nx = 1200;
	int ny = 600;
	// 设定每个block包含thread的数量为 8 x 8
	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny; //照片的像素数量
	size_t fb_size = 3 * num_pixels * sizeof(float); //一个帧缓冲的大小

	// 分配帧缓冲
	/* 为了进一步提高性能, 可以让GPU 把float变量转为8-bit的
	/ 再把数据读取回来，这样可以节省数据传输带宽*/
	float* fb;
	// 统一内存使用一个托管内存来共同管理host和device中的内存，并且自动在host和device中进行数据传输
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));

	clock_t start, stop;
	start = clock();
	// 在GPU上渲染帧缓冲
	dim3 blocks(nx / tx + 1, ny / ty + 1); // +1保证缓冲区(grid)不小于1200 x 800
	dim3 threads(tx, ty); //每个block的规格为 tx * ty
	render << <blocks, threads >> > (fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// 在CPU上把帧缓冲输出为ppm文件
	std::ofstream os("test.ppm");
	os << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			// 以下计算索引的方法原理同render中的注释
			size_t pixel_index = j * 3 * nx + i * 3;
			float r = fb[pixel_index + 0];
			float g = fb[pixel_index + 1];
			float b = fb[pixel_index + 2];
			int ir = int(255.99 * r);
			int ig = int(255.99 * g);
			int ib = int(255.99 * b);
			os << ir << " " << ig << " " << ib << "\n";
		}
	}
	os.close();
	std::cout << "Successfully save ppm image!" << std::endl;
	checkCudaErrors(cudaFree(fb));
	return 0;
}

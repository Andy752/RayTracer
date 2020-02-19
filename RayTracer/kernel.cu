#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "ray.h"

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

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
	if (hit_sphere(vec3(0, 0, -1), 0.5, r))
		return vec3(1, 0, 0);
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
	// i,j是当前thread所在grid中的唯一标识，以下计算是当grid和block二维时的计算方式
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return; //分配的总线程数超过了1200x800，超过的部分不需要参与计算
	int pixel_index = j * max_x + i; //pixel_index是把照片上(i,j)像素映射到fb中的下标变量
	// 利用当前线程计算照片中的像素(i,j)，分别计算r,g,b
	// 注意转换为float是很重要的，在GPU上float类型比double类型效率高
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	fb[pixel_index] = color(r);
}

int main() {
	// 渲染一张 1200 x 600 个像素的照片
	int nx = 1200;
	int ny = 600;
	// 设定每个block包含thread的数量为 8 x 8
	/* 设定为 8 x 8 的理由有两个：1.这样的一个较小的方形结构使得每个像素执行的工作量相似。
	假设在一个block中有一个像素执行的工作量比其他的大很多，那么这个block的效率会受限。
	我的理解是由于这些thread是并行的，所以每个block的效率取决于效率最低的thread。 
	2. 因为8x8=64，这是32的倍数。每个block的线程数量应为32的倍数 */
	int tx = 8;
	int ty = 8;
	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny; //照片的像素数量
	size_t fb_size = num_pixels * sizeof(vec3); //一个帧缓冲的大小

	// 分配帧缓冲
	/* 为了进一步提高性能, 可以让GPU 把float变量转为8-bit的
	/ 再把数据读取回来，这样可以节省数据传输带宽*/
	vec3* fb;
	// 统一内存使用一个托管内存来共同管理host和device中的内存，并且自动在host和device中进行数据传输
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));

	clock_t start, stop;
	start = clock();
	// 在GPU上渲染帧缓冲
	dim3 blocks(nx / tx + 1, ny / ty + 1); // +1保证缓冲区(grid)不小于1200 x 800
	dim3 threads(tx, ty); //每个block的规格为 tx * ty
	render << <blocks, threads >> > (fb, nx, ny,
		vec3(-2.0, -1.0, -1.0),
		vec3(4.0, 0.0, 0.0),
		vec3(0.0, 2.0, 0.0),
		vec3(0.0, 0.0, 0.0));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// 在CPU上把帧缓冲输出为ppm文件
	std::ofstream os("background.ppm");
	os << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			// 以下计算索引的方法原理同render中的注释
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			os << ir << " " << ig << " " << ib << "\n";
		}
	}
	os.close();
	std::cout << "Successfully save ppm image!" << std::endl;
	checkCudaErrors(cudaFree(fb));
	return 0;
}

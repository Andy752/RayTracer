﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <fstream>
#include <iostream>
#include <ctime>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include "../svpng/svpng.inc"

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "rt_texture.h"
#include "moving_sphere.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

// #define Sphereflake
#define K1 9
#define K2 81
#define K3 729
#define K4 6561
#define K5 59049

#ifdef  Sphereflake
const int sphere_num = 2 + K1 + K2 + K3 + K4;
#else
const int sphere_num = 22 * 22 + 1 + 3;
#endif

/* 要点：
1.在合适的地方添加 __device__ 标志
2.记得释放资源，如调用cudaFree
3.记得调整全部浮点数类型为float
4.合适地使用cuda的随机数
*/

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// 保证在退出前重置CUDA设备
		std::cerr << cudaGetErrorString(result) << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
	// i,j是当前thread所在grid中的唯一标识，以下计算是当grid和block二维时的计算方式
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return; // 分配的总线程数超过了1200x800，超过的部分不需要参与计算
	int pixel_index = j * max_x + i; // pixel_index是把照片上(i,j)像素映射到fb中的下标变量
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		// 利用当前线程计算照片中的像素(i,j)，分别计算r,g,b
		// 注意转换为float是很重要的，在GPU上float类型比double类型效率高
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny,
	curandState* rand_state
#ifdef Sphereflake
	, vec3* sphere_list_old, vec3* sphere_list_new, vec3* sphere_list_heart
#endif
) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		rt_texture *checker = new checker_texture(new constant_texture(vec3(0.2, 0.3, 0.1)),new constant_texture(vec3(0.9,0.9,0.9)));
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(checker)); // 平面大球
		int i = 1;
#ifndef  Sphereflake
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RND;
				vec3 center(a + 0.9f * RND, 0.2f, b + 0.9f * RND);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2f,
						new lambertian(new constant_texture(vec3(RND * RND, RND * RND, RND * RND))));
					//d_list[i++] = new moving_sphere(center, center + vec3(0.0f, 0.5f * RND, 0.0f), 0.0f, 1.0f, 0.2f,
					//	new lambertian(vec3(RND * RND, RND * RND, RND * RND))); // 注意相应修改free_world中的代码
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2f,
						new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
#endif

		* rand_state = local_rand_state;
#ifndef Sphereflake
		* d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);
		// * d_world = new bvh_node(d_list, 22 * 22 + 1 + 3,0.0f,1.0f,rand_state,0); // 把hitable_list改为bvh_node以使用bvh
#endif // !Sphereflake

#ifdef Sphereflake
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0)); // 主球
		//开始迭代创建3D科赫雪花球
		int level = 4; // 计算4层，最后一层有Seven_4个球
		vec3 h0(0, 1, 0);
		vec3 last_heart(0, 0, 0);
		sphere_list_old[0] = h0;
		sphere_list_heart[0] = last_heart;
		for (int k = 0; k < level; ++k) {
			// 本轮迭代需要创建 9 * sphere_num_level数量的球体，即需要sphere_num_level个上一个level的球心
			int sphere_num_level = (k == 0 ? 1 : (k == 1 ? K1 : (k == 2 ? K2 : (k == 3 ? K3 : (k == 4 ? K4 : K5)))));
			for (int j = 0; j < sphere_num_level; ++j) {
				float last_r = powf(1.0f / 3.0f, k);
				float current_r = last_r / 3.0f;
				// 取得父亲球体的上法向量
				vec3 up = sphere_list_old[j] - sphere_list_heart[int(j / 9)];
				/* DEBUG
				printf("****** sphere_num_level = %d, j = %d, j/t = %d , *****\n", sphere_num_level, j, int(j / 7));
				printf("sphere_list_old[j] = (%f,%f,%f),sphere_list_heart[int(j/9)]=(%f,%f,%f)", sphere_list_old[j].x(),
					sphere_list_old[j].y(), sphere_list_old[j].z(), sphere_list_heart[int(j / 9)].x(),
					sphere_list_heart[int(j / 9)].y(), sphere_list_heart[int(j / 9)].z());
				*/
				up.make_unit_vector();
				//vec3 h1 = sphere_list_old[j] + (last_r + current_r) * up;
				// 绘制周围6个小球
				// 构造与up垂直的一个向量n1
				vec3 n1;
				if (fabs(up.y()) > FLT_EPSILON) {
					n1 = vec3(1.0f, -up.x() / up.y(), 0);
				}
				else if (fabs(up.x()) > FLT_EPSILON) {
					n1 = vec3(-up.y() / up.x(), 1.0f, 0);
				}
				else {
					n1 = vec3(1.0f, 0, -up.x() / up.z());
				}
				n1.make_unit_vector(); // 周围第一个小球的上法向量
				vec3 h1 = sphere_list_old[j] + (last_r + current_r) * n1; // 周围第一个小球球心
				// 旋转60度得到周围第二个小球的上法向量
				const float theta = 3.14159265f / 3.0f; // pi/3
				vec3 n2 = n1 * cos(theta) + cross(n1, up) * sin(theta);
				n2.make_unit_vector();
				vec3 h2 = sphere_list_old[j] + (last_r + current_r) * n2; // 周围第二个小球球心
				// 周围第三个小球球心
				vec3 n3 = n2 * cos(theta) + cross(n2, up) * sin(theta);
				n3.make_unit_vector();
				vec3 h3 = sphere_list_old[j] + (last_r + current_r) * n3;
				// 周围第四个小球球心
				vec3 n4 = n3 * cos(theta) + cross(n3, up) * sin(theta);
				n4.make_unit_vector();
				vec3 h4 = sphere_list_old[j] + (last_r + current_r) * n4;
				// 周围第五个小球球心
				vec3 n5 = n4 * cos(theta) + cross(n4, up) * sin(theta);
				n5.make_unit_vector();
				vec3 h5 = sphere_list_old[j] + (last_r + current_r) * n5;
				// 周围第六个小球球心
				vec3 n6 = n5 * cos(theta) + cross(n5, up) * sin(theta);
				n6.make_unit_vector();
				vec3 h6 = sphere_list_old[j] + (last_r + current_r) * n6;
				// 求顶部3个小球球心
				vec3 n7 = 1.5f * up + n1;
				n7.make_unit_vector();
				vec3 h7 = sphere_list_old[j] + (last_r + current_r) * n7;// 顶部第一个小球球心
				vec3 n8 = 1.5f * up + n3;
				n8.make_unit_vector();
				vec3 h8 = sphere_list_old[j] + (last_r + current_r) * n8;// 顶部第二个小球球心
				vec3 n9 = 1.5f * up + n5;
				n9.make_unit_vector();
				vec3 h9 = sphere_list_old[j] + (last_r + current_r) * n9;// 顶部第三个小球球心

				/* DEBUG
				printf("up = (%f,%f,%f)\n", up.x(), up.y(), up.z());
				printf("h1 = (%f,%f,%f)\n", h1.x(), h1.y(), h1.z());

				printf("n1 = (%f,%f,%f)\n", n1.x(), n1.y(), n1.z());
				printf("h2 = (%f,%f,%f)\n", h2.x(), h2.y(), h2.z());

				printf("n2 = (%f,%f,%f)\n", n2.x(), n2.y(), n2.z());
				printf("h3 = (%f,%f,%f)\n", h3.x(), h3.y(), h3.z());

				printf("n3 = (%f,%f,%f)\n", n3.x(), n3.y(), n3.z());
				printf("h4 = (%f,%f,%f)\n", h4.x(), h4.y(), h4.z());

				printf("n4 = (%f,%f,%f)\n", n4.x(), n4.y(), n4.z());
				printf("h5 = (%f,%f,%f)\n", h5.x(), h5.y(), h5.z());

				printf("n5 = (%f,%f,%f)\n", n5.x(), n5.y(), n5.z());
				printf("h6 = (%f,%f,%f)\n", h6.x(), h6.y(), h6.z());

				printf("n6 = (%f,%f,%f)\n", n6.x(), n6.y(), n6.z());
				printf("h7 = (%f,%f,%f)\n", h7.x(), h7.y(), h7.z());
				printf("-------------------------------------------\n");
				*/

				// 添加周围小球到列表中
				d_list[i++] = new sphere(h1, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h2, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h3, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h4, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h5, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h6, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h7, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h8, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				d_list[i++] = new sphere(h9, current_r, new metal(vec3(0.3f, 0.3f, 0.3f), 0.0));
				sphere_list_new[j * 9] = h1;
				sphere_list_new[j * 9 + 1] = h2;
				sphere_list_new[j * 9 + 2] = h3;
				sphere_list_new[j * 9 + 3] = h4;
				sphere_list_new[j * 9 + 4] = h5;
				sphere_list_new[j * 9 + 5] = h6;
				sphere_list_new[j * 9 + 6] = h7;
				sphere_list_new[j * 9 + 7] = h8;
				sphere_list_new[j * 9 + 8] = h9;
			}
			// 交换三个球心列表
			vec3* temp = sphere_list_heart;
			sphere_list_heart = sphere_list_old;
			sphere_list_old = sphere_list_new;
			sphere_list_new = temp;
		}
		*d_world = new hitable_list(d_list, sphere_num);
#endif // Sphereflake

		vec3 lookfrom(8, 8, 3);
		vec3 lookat(0, 1, 0);
		float dist_to_focus = 10.0;
		float aperture = 0.0;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			20.0f,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0f,
			1.0f);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	for (int i = 0; i < sphere_num; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		//delete ((moving_sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

void write_to_png(char const* filename, int w, int h, int n, vec3* data) {
	unsigned char* uc_data = (unsigned char*)std::malloc(w * h * n * sizeof(unsigned char));
	unsigned char* p = uc_data;
	FILE* fp = fopen(filename, "wb");
	for (int j = h - 1; j >= 0; j--) {
		for (int i = 0; i < w; i++) {
			size_t pixel_index = j * w + i;
			*p++ = (unsigned char)(255.99 * data[pixel_index].r());    /* R */
			*p++ = (unsigned char)(255.99 * data[pixel_index].g());    /* G */
			*p++ = (unsigned char)(255.99 * data[pixel_index].b());    /* B */
		}
	}
	svpng(fp, w, h, uc_data, 0);
	fclose(fp);
	std::free(uc_data);
}

void write_to_ppm(char const* filename, int w, int h, vec3* data) {
	std::ofstream os(filename);
	os << "P3\n" << w << " " << h << "\n255\n";
	for (int j = h - 1; j >= 0; j--) {
		for (int i = 0; i < w; i++) {
			size_t pixel_index = j * w + i;
			int ir = int(255.99 * data[pixel_index].r());
			int ig = int(255.99 * data[pixel_index].g());
			int ib = int(255.99 * data[pixel_index].b());
			os << ir << " " << ig << " " << ib << "\n";
		}
	}
	os.close();
}

int main() {
	int nx = 1024; //1024
	int ny = 576; //576
	int ns = 100;
	// 设定每个block包含thread的数量为 8 x 8
	/* 设定为 8 x 8 的理由有两个：1.这样的一个较小的方形结构使得每个像素执行的工作量相似。
	假设在一个block中有一个像素执行的工作量比其他的大很多，那么这个block的效率会受限。
	我的理解是由于这些thread是并行的，所以每个block的效率取决于效率最低的thread。
	2. 因为8x8=64，这是32的倍数。每个block的线程数量应为32的倍数 */
	int tx = 8;
	int ty = 8;

	std::cerr << "渲染一张大小为 " << nx << "x" << ny << " 的照片。每个像素采集" << ns << "个样本。";
	std::cerr << "CUDA块大小为" << tx << "x" << ty << "。\n";

	int num_pixels = nx * ny; // 照片的像素数量
	size_t fb_size = num_pixels * sizeof(vec3); // 一个帧缓冲的大小

	// 分配帧缓冲
	/* 为了进一步提高性能, 可以让GPU 把float变量转为8-bit的
	/ 再把数据读取回来，这样可以节省数据传输带宽*/
	vec3* fb;
	// 统一内存使用一个托管内存来共同管理host和device中的内存，并且自动在host和device中进行数据传输
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));

	// 分配random state，用于在GPU上生成随机数。
	// 前缀d_表示这些数据仅在GPU上使用。
	// cudaMalloc分配显卡上的内存
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)& d_rand_state, num_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)& d_rand_state2, sphere_num * sizeof(curandState)));

	// 初始化d_rand_state2，用于创建world
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef Sphereflake
	vec3* sphere_list_old;
	checkCudaErrors(cudaMallocManaged((void**)& sphere_list_old, K4));
	vec3* sphere_list_new;
	checkCudaErrors(cudaMallocManaged((void**)& sphere_list_new, K4));
	vec3* sphere_list_heart;
	checkCudaErrors(cudaMallocManaged((void**)& sphere_list_heart, K4));

#endif // Sphereflake

	// make our world of hitables & the camera
	hitable** d_list;
	int num_hitables = sphere_num;
	checkCudaErrors(cudaMalloc((void**)& d_list, num_hitables * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)& d_world, sizeof(hitable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)& d_camera, sizeof(camera*)));

#ifdef Sphereflake
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2,sphere_list_old,sphere_list_new,sphere_list_heart);
#else
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
#endif
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();
	// 在GPU上渲染帧缓冲
	dim3 blocks(nx / tx + 1, ny / ty + 1); // +1保证缓冲区(grid)不小于1200 x 800
	dim3 threads(tx, ty); //每个block的规格为 tx * ty
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "渲染耗时：" << timer_seconds << " 秒。\n";

	// Output FB as Image
	//write_to_ppm("result/Motion_Blur.ppm", nx, ny, fb);
	write_to_png("result/check_texture.png", nx, ny, 3, fb);

	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_rand_state2));
	checkCudaErrors(cudaFree(fb));

#ifdef Sphereflake
	checkCudaErrors(cudaFree(sphere_list_old));
	checkCudaErrors(cudaFree(sphere_list_new));
	checkCudaErrors(cudaFree(sphere_list_heart));
#endif // Sphereflake

	cudaDeviceReset();
	return 0;
}
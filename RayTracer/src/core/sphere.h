#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "hitable.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	// __device__ virtual bool sphere::bounding_box(float t0, float t1, aabb& box) const;
	vec3 center;
	float radius;
	material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

// __device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const
// {
// 	box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
// 	return true;
// }
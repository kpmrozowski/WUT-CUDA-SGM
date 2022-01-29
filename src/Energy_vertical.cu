#include "Energy.hpp"
#include <climits>
#include <stdint.h>

namespace sgm {

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_vertical_path_kernel(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp)
{
	static_assert(DIRECTION == 1 || DIRECTION == -1, "");
	if(width == 0 || height == 0){
		return;
	}
	const unsigned int d = threadIdx.z;
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (d > MAX_DISPARITY - 1 || x > width - 1) return;
	int y;
	for (y = (DIRECTION==1 ? 1 : height-1); (DIRECTION==1 ? height-y : y+1) > 0; y+=DIRECTION) {
		unsigned int idx_p___d_0 = x + y * width + d * width * height;
		unsigned int idx_pmr_dm1 = x + (y - DIRECTION) * width + (d - 1u) * width * height;
		unsigned int idx_pmr_d_0 = x + (y - DIRECTION) * width + (d + 0u) * width * height;
		unsigned int idx_pmr_dp1 = x + (y - DIRECTION) * width + (d + 1u) * width * height;
		cost_type C = cost_in[idx_p___d_0];
		cost_type Ldm1 = UINT8_MAX;
		cost_type Ld_0 = UINT8_MAX;
		cost_type Ldp1 = UINT8_MAX;
		cost_type minL = UINT8_MAX;
		C = cost_in[idx_p___d_0];
		if (d != 0) 				Ldm1 = dest[idx_pmr_dm1];
									Ld_0 = dest[idx_pmr_d_0];
		if (d != MAX_DISPARITY - 1) Ldp1 = dest[idx_pmr_dp1];
		for (int di = 0; di < MAX_DISPARITY; ++di) {
			int idx = x + (y - DIRECTION) * width + di * width * height;
			if (dest[idx] < minL) minL = dest[idx];
		}
		int result = C + min(min(
			Ld_0, 
			Ldp1 + static_cast<cost_type>(p1)), min(
			Ldm1 + static_cast<cost_type>(p1), 
			minL + static_cast<cost_type>(p2)))
			 - minL;
		// int result = minL;
		if (result > UINT8_MAX) {
			dest[idx_p___d_0] = static_cast<cost_type>(UINT8_MAX);
			printf("overflow: result=%d\n", result);
			__syncthreads();
			continue;
		}
		dest[idx_p___d_0] = static_cast<cost_type>(result);
		__syncthreads();
	}

	// if (DIRECTION == 1) {
	// 	// dest[x + 0 + d * width * height] = 0;
	// 	for (int y = 1; y < height; ++y) {
	// 		unsigned int idx_p___d_0 = x + y * width + d * width * height;
	// 		unsigned int idx_pmr_dm1 = x + (y - DIRECTION) * width + (d - 1) * width * height;
	// 		unsigned int idx_pmr_d_0 = x + (y - DIRECTION) * width + (d + 0) * width * height;
	// 		unsigned int idx_pmr_dp1 = x + (y - DIRECTION) * width + (d + 1) * width * height;
	// 		cost_type C = cost_in[idx_p___d_0];
	// 		cost_type Ldm1 = UINT8_MAX;
	// 		cost_type Ld_0 = UINT8_MAX;
	// 		cost_type Ldp1 = UINT8_MAX;
	// 		cost_type minL = UINT8_MAX;
	// 		C = cost_in[idx_p___d_0];
	// 		if (d != 0) 				Ldm1 = dest[idx_pmr_dm1];
	// 									Ld_0 = dest[idx_pmr_d_0];
	// 		if (d != MAX_DISPARITY - 1) Ldp1 = dest[idx_pmr_dp1];
	// 		for (int di = 0; di < MAX_DISPARITY; ++di) {
	// 			int idx = x + (y - DIRECTION) * width + di * width * height;
	// 			if (dest[idx] < minL) minL = dest[idx];
	// 		}
	// 		int result = C + min(min(
	// 			Ld_0, 
	// 			Ldp1 + static_cast<cost_type>(p1)), min(
	// 			Ldm1 + static_cast<cost_type>(p1), 
	// 			minL + static_cast<cost_type>(p2)))
	// 			 - minL;
	// 		// int result = minL;
	// 		if (result > UINT8_MAX) {
	// 			dest[idx_p___d_0] = static_cast<cost_type>(UINT8_MAX);
	// 			printf("overflow: result=%d\n", result);
	// 			__syncthreads();
	// 			continue;
	// 		}
	// 		dest[idx_p___d_0] = static_cast<cost_type>(result);
	// 		__syncthreads();
	// 	}
	// }
	// else {
	// 	dest[x + (height - 1) * width + d * width * height] = 0;
	// 	for (int y = height - 2; y > -1; --y) {
	// 		unsigned int idx_p___d_0 = x + y * width + d * width * height;
	// 		unsigned int idx_pmr_dm1 = x + (y - DIRECTION) * width + (d - 1) * width * height;
	// 		unsigned int idx_pmr_d_0 = x + (y - DIRECTION) * width + (d + 0) * width * height;
	// 		unsigned int idx_pmr_dp1 = x + (y - DIRECTION) * width + (d + 1) * width * height;
	// 		cost_type C = cost_in[idx_p___d_0];
	// 		cost_type Ldm1 = UINT8_MAX;
	// 		cost_type Ld_0 = UINT8_MAX;
	// 		cost_type Ldp1 = UINT8_MAX;
	// 		cost_type minL = UINT8_MAX;
	// 		C = cost_in[idx_p___d_0];
	// 		if (d != 0) 				Ldm1 = dest[idx_pmr_dm1];
	// 									Ld_0 = dest[idx_pmr_d_0];
	// 		if (d != MAX_DISPARITY - 1) Ldp1 = dest[idx_pmr_dp1];
	// 		for (int di = 0; di < MAX_DISPARITY; ++di) {
	// 			int idx = x + (y - DIRECTION) * width + di * width * height;
	// 			if (dest[idx] < minL) minL = dest[idx];
	// 		}
	// 		int result = C + min(min(
	// 			Ld_0, 
	// 			Ldp1 + static_cast<cost_type>(p1)), min(
	// 			Ldm1 + static_cast<cost_type>(p1), 
	// 			minL + static_cast<cost_type>(p2)))
	// 			 - minL;
	// 		// int result = minL;
	// 		if (result > UINT8_MAX) {
	// 			dest[idx_p___d_0] = static_cast<cost_type>(UINT8_MAX);
	// 			printf("overflow: result=%d\n", result);
	// 			__syncthreads();
	// 			continue;
	// 		}
	// 		dest[idx_p___d_0] = static_cast<cost_type>(result);
	// 		__syncthreads();
	// 	}
	// }
}

template <unsigned int MAX_DISPARITY>
void compute_energy_up2down(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_up2down\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_x_dim = num_threads / block_d_dim;
	int grid_dim = width / block_x_dim;
	if (grid_dim * block_x_dim < width) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	aggregate_vertical_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_down2up(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_down2up\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_x_dim = num_threads / block_d_dim;
	int grid_dim = width / block_x_dim;
	if (grid_dim * block_x_dim < width) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
#ifdef DEBUG
	int y0=3, offset=0, start=offset, end=48+offset, dsize = end-start;
    cost_type lookup[height * width * dsize];
    // cost_type lookup_in[height * width * dsize];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest+start, sizeof(cost_type) * height * width * dsize, cudaMemcpyDeviceToHost);
    // cudaMemcpy(lookup_in, cost_in+start, sizeof(cost_type) * height * width * dsize, cudaMemcpyDeviceToHost);
    cost_type max = 0;
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int d = start; d < end; ++d) {
                cost_type val = lookup[x + y * width + (d-start) * width * height];
                if (max < val && val < UINT8_MAX) max = val;
            }
        }
    }
    printf("max=%d\n", max);
    for (int x = 0; x < width; ++x) {
        printf("x:%3.3d", x);
        for (int d = start; d < end; ++d) {
            cost_type val = lookup[x + y0 * width + (d-start) * width * height];
            // cost_type val2 = lookup_in[x + y0 * width + (d-start) * width * height];
            // printf("cost[%d][300][%d]=%d\t", x, d, val);
            printf("%4.1d", val);
            // printf("%3.1d", val - val2);
        }
        printf("\n");
    }
#endif
	printf("My compute_energy_down2up ends\n");
}

template void compute_energy_up2down<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_up2down<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_up2down<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_down2up<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_down2up<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_down2up<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}



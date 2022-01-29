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
	if(width == 0 || height == 0){
		return;
	}
	const unsigned int d = threadIdx.z;
	const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (d > MAX_DISPARITY || x > width - 1) return;
	
	if (DIRECTION == 1) {
		dest[x + 0 + d * width * height] = 0;
		for (int y = 1; y < height; ++y) {
			unsigned int idx_p___d_0 = x + y * width + d * width * height;
			unsigned int idx_pmr_dm1 = x + (y - DIRECTION) * width + (d - 1) * width * height;
			unsigned int idx_pmr_d_0 = x + (y - DIRECTION) * width + (d + 0) * width * height;
			unsigned int idx_pmr_dp1 = x + (y - DIRECTION) * width + (d + 1) * width * height;
			
			int C = cost_in[idx_p___d_0];
			int Ld_0 = dest[idx_pmr_d_0];
			int Ldp1 = dest[idx_pmr_dp1];
			int Ldm1 = dest[idx_pmr_dm1];
			int minL = INT_MAX;
			for (int i = 0; i < blockDim.z; ++i) {
				int idx = x + y * width + i * width * height;
				if (dest[idx] < minL) minL = dest[idx];
			}
			dest[idx_p___d_0] = C + min(min(Ld_0, Ldp1 + p1), min(Ldm1 + p1, minL + p2)) - minL;
			__syncthreads();
		}
	} else {
		dest[x + (height - 1) * width + d * width * height] = 0;
		for (int y = height - 2; y > -1; --y) {
			unsigned int idx_p___d_0 = x + y * width + d * width * height;
			unsigned int idx_pmr_dm1 = x + (y - DIRECTION) * width + (d - 1) * width * height;
			unsigned int idx_pmr_d_0 = x + (y - DIRECTION) * width + (d + 0) * width * height;
			unsigned int idx_pmr_dp1 = x + (y - DIRECTION) * width + (d + 1) * width * height;
			
			int C = cost_in[idx_p___d_0];
			int Ld_0 = dest[idx_pmr_d_0];
			int Ldp1 = dest[idx_pmr_dp1];
			int Ldm1 = dest[idx_pmr_dm1];
			int minL = INT_MAX;
			for (int i = 0; i < blockDim.z; ++i) {
				int idx = x + y * width + i * width * height;
				if (dest[idx] < minL) minL = dest[idx];
			}
			dest[idx_p___d_0] = C + min(min(Ld_0, Ldp1 + p1), min(Ldm1 + p1, minL)) - minL;
			__syncthreads();
		}
	}
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
#ifdef DEBUG
    cost_type lookup[height * width * 16];
    cost_type lookup_in[height * width * 16];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest, sizeof(cost_type) * height * width * 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(lookup_in, cost_in, sizeof(cost_type) * height * width * 16, cudaMemcpyDeviceToHost);
    cost_type max = 0;
    for (int x = 10; x < width - 10; ++x) {
        for (int y = 10; y < height - 10; ++y) {
            for (int d = 0; d < 16; ++d) {
                cost_type val = lookup[x + y * width + d * width * height];
                if (max < val && val < 100) max = val;
            }
        }
    }
    printf("max=%d\n", max);
    for (int x = 10; x < width - 10; ++x) {
        printf("x:%3.2d", x);
        for (int d = 0; d < 16; ++d) {
            cost_type val = lookup[x + 9 * width + d * width * height];
            cost_type val2 = lookup_in[x + 9 * width + d * width * height];
            // printf("cost[%d][300][%d]=%d\t", x, d, val);
            printf("%3.1d", val - val2);
        }
        printf("\n");
    }
#endif
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
	const int gdim = 1;
	const int bdim = width;
	aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
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



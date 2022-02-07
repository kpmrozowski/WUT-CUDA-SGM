#include "Energy.hpp"

namespace sgm {

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
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
	const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (d > MAX_DISPARITY - 1 || y > height - 1) return;
	int x;
	for (x = (DIRECTION==1 ? 1 : width-1); (DIRECTION==1 ? width-x : x+1) > 0; x+=DIRECTION) {
		unsigned int idx_p___d_0 = x + y * width + d * width * height;
		unsigned int idx_pmr_dm1 = (x - DIRECTION) + y * width + (d - 1u) * width * height;
		unsigned int idx_pmr_d_0 = (x - DIRECTION) + y * width + (d + 0u) * width * height;
		unsigned int idx_pmr_dp1 = (x - DIRECTION) + y * width + (d + 1u) * width * height;
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
			int idx = (x - DIRECTION) + y * width + di * width * height;
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
}

template <unsigned int MAX_DISPARITY>
void compute_energy_L2R(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	// printf("My compute_energy_L2R\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_y_dim = num_threads / block_d_dim;
	int grid_dim = height / block_y_dim;
	if (grid_dim * block_y_dim < height) {
		grid_dim += 1;
	}
	// const dim3 bdim(1, block_y_dim, block_d_dim);
	// const dim3 gdim(1, grid_dim, 1);
	const dim3 bdim(1, 1, block_d_dim);
	const dim3 gdim(1, height, 1);
	aggregate_horizontal_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void compute_energy_R2L(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	// printf("My compute_energy_R2L\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_y_dim = num_threads / block_d_dim;
	int grid_dim = height / block_y_dim;
	if (grid_dim * block_y_dim < height) {
		grid_dim += 1;
	}
	// const dim3 bdim(1, block_y_dim, block_d_dim);
	// const dim3 gdim(1, grid_dim, 1);
	const dim3 bdim(1, 1, block_d_dim);
	const dim3 gdim(1, height, 1);
	aggregate_horizontal_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
#ifdef DEBUG
	int x0=50, offset=0, start=offset, end=48+offset, dsize = end-start;
    cost_type lookup[height * width * dsize];
    // cost_type lookup_in[height * width * dsize];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest+start * height * width, sizeof(cost_type) * height * width * dsize, cudaMemcpyDeviceToHost);
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
    for (int y = 0; y < height; ++y) {
        printf("y:%3.3d", y);
        for (int d = start; d < end; ++d) {
            cost_type val = lookup[x0 + y * width + (d-start) * width * height];
            // cost_type val2 = lookup_in[x + y0 * width + (d-start) * width * height];
            // printf("cost[%d][300][%d]=%d\t", x, d, val);
            printf("%4.1d", val);
            // printf("%3.1d", val - val2);
        }
        printf("\n");
    }
#endif
	// printf("My compute_energy_R2L ends\n");
}


template void compute_energy_L2R<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_L2R<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_L2R<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_R2L<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}
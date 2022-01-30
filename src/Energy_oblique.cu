#include "Energy.hpp"

namespace sgm {

template <int DIRECTION_X, int DIRECTION_Y, unsigned int MAX_DISPARITY>
__global__ void aggregate_oblique_path_kernel(
	cost_type *dest,
	const cost_type *cost_in,
	const int width,
	const int height,
	const unsigned int p1,
	const unsigned int p2,
	const int min_disp)
{
	static_assert(DIRECTION_X == 1 || DIRECTION_X == -1, "");
	static_assert(DIRECTION_Y == 1 || DIRECTION_Y == -1, "");
	if(width == 0 || height == 0){
		return;
	}
	const unsigned int d = threadIdx.z;
	const unsigned int xy = threadIdx.x + blockIdx.x * blockDim.x;
	if (d > MAX_DISPARITY - 1 || xy >  width + height - 1) return;
	const int x0
	 = DIRECTION_X == 1 && DIRECTION_Y == 1 ? 1 - ((height - 1) / 2)
	 : DIRECTION_X ==-1 && DIRECTION_Y ==-1 ? width + (height - 1) / 2 + (width + height) % 2 - 3
	 : DIRECTION_X == 1 && DIRECTION_Y ==-1 ? 1 - ((height - 1) / 2)
	 : DIRECTION_X ==-1 && DIRECTION_Y == 1 ? width + (height - 1) / 2 + (width + height) % 2 - 3
	 : 0;
	// const int y0 = height / 2 + 1;
	const int y0
	 = DIRECTION_X == 1 && DIRECTION_Y == 1 ? height / 2 + 1
	 : DIRECTION_X ==-1 && DIRECTION_Y ==-1 ? height - height / 2 - (width + height + 1) % 2 - 2
	 : DIRECTION_X == 1 && DIRECTION_Y ==-1 ? height - height / 2 - (width + height + 1) % 2 - 2
	 : DIRECTION_X ==-1 && DIRECTION_Y == 1 ? height / 2 + 1
	 : 0;
	const int u = DIRECTION_X * ((xy + ( height       % 2)) / 2);
	const int v = DIRECTION_Y * ((xy + ((height + 1)  % 2)) / 2);
	const int steps_end = (width + height - 1) / 2 - 1;
	int x, y;
	for (int step = 0; step < steps_end; ++step) {
		x = x0 + u + DIRECTION_X * step;
		y = y0 - v + DIRECTION_Y * step;
		if (DIRECTION_X == 1 && DIRECTION_Y == 1 && (1 > x || x > width - 1 || 1 > y || y > height - 1)) {
			__syncthreads();
			continue;
		}
		if (DIRECTION_X ==-1 && DIRECTION_Y ==-1 && (0 > x || x > width - 2 || 0 > y || y > height - 2)) {
			__syncthreads();
			continue;
		}
		if (DIRECTION_X == 1 && DIRECTION_Y ==-1 && (1 > x || x > width - 1 || 0 > y || y > height - 2)) {
			__syncthreads();
			continue;
		}
		if (DIRECTION_X ==-1 && DIRECTION_Y == 1 && (0 > x || x > width - 2 || 1 > y || y > height - 1)) {
			__syncthreads();
			continue;
		}
		unsigned int idx_p___d_0 = x + y * width + d * width * height;
		unsigned int idx_pmr_dm1 = (x - DIRECTION_X) + (y - DIRECTION_Y) * width + (d - 1u) * width * height;
		unsigned int idx_pmr_d_0 = (x - DIRECTION_X) + (y - DIRECTION_Y) * width + (d + 0u) * width * height;
		unsigned int idx_pmr_dp1 = (x - DIRECTION_X) + (y - DIRECTION_Y) * width + (d + 1u) * width * height;
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
			int idx = (x - DIRECTION_X) + (y - DIRECTION_Y) * width + di * width * height;
			if (dest[idx] < minL) minL = dest[idx];
		}
		int result = C + min(min(
			Ld_0, 
			Ldp1 + static_cast<cost_type>(p1)), min(
			Ldm1 + static_cast<cost_type>(p1), 
			minL + static_cast<cost_type>(p2)))
			 - minL;
		// int result = minL;
		// int result = step % INT8_MAX;//-x0 + u;blockIdx.x
		// int result = blockIdx.x % INT8_MAX;
		if (result > UINT8_MAX) {
			dest[idx_p___d_0] = static_cast<cost_type>(UINT8_MAX);
			// printf("overflow: result=%d\n", result);
			__syncthreads();
			continue;
		}
		dest[idx_p___d_0] = static_cast<cost_type>(result);
		__syncthreads();
	}
}

template <unsigned int MAX_DISPARITY>
void compute_energy_upL2downR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_upL2downR\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_xydim = num_threads / block_d_dim;
	int grid_dim = (height + width) / block_xydim;
	float a = grid_dim * block_xydim;
	float b = (height + width);
	if (a < b) {
		grid_dim += 1;
	}
	const dim3 bdim(block_xydim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	// const dim3 bdim(1, 1, block_d_dim);
	// const dim3 gdim(height + width - 2, 1, 1);
	aggregate_oblique_path_kernel<1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
// #ifdef DEBUG
// 	int y0=373, offset=0, start=offset, end=48+offset, dsize = end-start;
//     cost_type lookup[height * width * dsize];
//     // cost_type lookup_in[height * width * dsize];
//     printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
//     cudaMemcpy(lookup, dest+start * height * width, sizeof(cost_type) * height * width * dsize, cudaMemcpyDeviceToHost);
//     // cudaMemcpy(lookup_in, cost_in+start, sizeof(cost_type) * height * width * dsize, cudaMemcpyDeviceToHost);
//     cost_type max = 0;
//     for (int x = 0; x < width; ++x) {
//         for (int y = 0; y < height; ++y) {
//             for (int d = start; d < end; ++d) {
//                 cost_type val = lookup[x + y * width + (d-start) * width * height];
//                 if (max < val && val < static_cast<cost_type>(UINT8_MAX)) max = val;
//             }
//         }
//     }
//     printf("max=%d\n", max);
//     for (int x = 0; x < width; ++x) {
//         printf("x:%3.3d", x);
//         for (int d = start; d < end; ++d) {
//             cost_type val = lookup[x + y0 * width + (d-start) * width * height];
//             // cost_type val2 = lookup_in[x + y0 * width + (d-start) * width * height];
//             // printf("cost[%d][300][%d]=%d\t", x, d, val);
//             printf("%4.1d", val);
//             // printf("%3.1d", val - val2);
//         }
//         printf("\n");
//     }
// 	// int d0 = 5;
//     // for (int y = 0; y < height; ++y) {
//     //     printf("y:%3.3d", y);
//     //     for (int x = 0; x < width; ++x) {
//     //         cost_type val = lookup[x + y * width + (d0-start) * width * height];
//     //         // cost_type val2 = lookup_in[x + y0 * width + (d-start) * width * height];
//     //         // printf("cost[%d][300][%d]=%d\t", x, d, val);
//     //         printf("%4.1d", val);
//     //         // printf("%3.1d", val - val2);
//     //     }
//     //     printf("\n");
//     // }
// #endif
	printf("My compute_energy_upL2downR end\n");
}

template <unsigned int MAX_DISPARITY>
void compute_energy_upR2downL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_upR2downL\n");
	cudaMemcpy(dest, cost_in, sizeof(cost_type) * height * width * MAX_DISPARITY, cudaMemcpyDeviceToDevice);
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_xydim = num_threads / block_d_dim;
	int grid_dim = (height + width) / block_xydim;
	float a = grid_dim * block_xydim;
	float b = (height + width);
	if (a < b) {
		grid_dim += 1;
	}
	const dim3 bdim(block_xydim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	aggregate_oblique_path_kernel<-1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
	printf("My compute_energy_upR2downL end\n");
}

template <unsigned int MAX_DISPARITY>
void compute_energy_downR2upL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_downR2upL\n");
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_xydim = num_threads / block_d_dim;
	int grid_dim = (height + width) / block_xydim;
	float a = grid_dim * block_xydim;
	float b = (height + width);
	if (a < b) {
		grid_dim += 1;
	}
	const dim3 bdim(block_xydim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	aggregate_oblique_path_kernel<-1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
	printf("My compute_energy_downR2upL end\n");
}

template <unsigned int MAX_DISPARITY>
void compute_energy_downL2upR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	printf("My compute_energy_downL2upR\n");
    int num_threads = 1024;
	int block_d_dim = nextPowerOf2(MAX_DISPARITY);
	int block_xydim = num_threads / block_d_dim;
	int grid_dim = (height + width) / block_xydim;
	float a = grid_dim * block_xydim;
	float b = (height + width);
	if (a < b) {
		grid_dim += 1;
	}
	const dim3 bdim(block_xydim, 1, block_d_dim);
	const dim3 gdim(grid_dim, 1, 1);
	aggregate_oblique_path_kernel<1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, cost_in, width, height, p1, p2, min_disp);
	printf("My compute_energy_downL2upR end\n");
}


template void compute_energy_upL2downR<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upL2downR<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upL2downR<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_upR2downL<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downR2upL<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<64u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<128u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template void compute_energy_downL2upR<256u>(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

}
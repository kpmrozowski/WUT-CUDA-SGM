#include "Energy.hpp"
#include <stdint.h>

namespace sgm {

template <unsigned int MAX_DISPARITY>
__global__ void sum_energy_kernel(
	cost_sum_type *dest,
	const cost_type *buffer_in,
	int width,
	int height,
	int num_paths)
{
    const unsigned int width_height = width * height;
    const unsigned int buffer_step = width * height * MAX_DISPARITY;
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y;
    const unsigned int d = blockIdx.z;
    if (x > width -1 || y > height - 1) return;
    cost_sum_type sum = 0;
    for (unsigned int path = 0; path < num_paths; ++path) {
        sum += buffer_in[x + y * width + d * width_height + path * buffer_step];
    }
    dest[x + y * width + d * width_height] = sum;
    // for (unsigned int d = 0; d < MAX_DISPARITY; ++d) {
    //     cost_sum_type sum = 0;
    //     for (unsigned int path = 0; path < num_paths; ++path) {
    //         sum += buffer_in[x + y * width + d * width_height + path * buffer_step];
    //     }
    //     dest[x + y * width + d * width_height] = sum;
    // }
    
}


template <unsigned int MAX_DISPARITY>
void sum_energy_all_paths(
	cost_sum_type *dest,
	const cost_type *buffer_in,
	int width,
	int height,
	int num_paths,
	cudaStream_t stream)
{
	printf("My sum_energy_all_paths\n");
    int num_threads = 1024;
	int block_y_dim = nextPowerOf2(height); // 512
	int block_x_dim = num_threads / block_y_dim; // 2
	int grid_dim = width / block_x_dim; // 238
	if (grid_dim * block_x_dim < width) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, block_y_dim, 1);
	const dim3 gdim(grid_dim, 1, MAX_DISPARITY);
	sum_energy_kernel<MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
		dest, buffer_in, width, height, num_paths);
#ifdef DEBUG
	int y0=3, offset=0, start=offset, end=16+offset, dsize = end-start;
    cost_sum_type lookup[height * width * dsize];
    // cost_type lookup_in[height * width * dsize];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest+start * height * width, sizeof(cost_sum_type) * height * width * dsize, cudaMemcpyDeviceToHost);
    cost_sum_type max = 0;
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            for (int d = start; d < end; ++d) {
                cost_sum_type val = lookup[x + y * width + (d-start) * width * height];
                if (max < val && val < UINT16_MAX) max = val;
            }
        }
    }
    printf("max=%d\n", max);
    for (int x = 0; x < width; ++x) {
        printf("x:%3.3d", x);
        for (int d = start; d < end; ++d) {
            cost_sum_type val = lookup[x + y0 * width + (d-start) * width * height];
            printf("%4.1d", val);
        }
        printf("\n");
    }
#endif
	printf("My sum_energy_all_paths ends\n");
}

template void sum_energy_all_paths<64u>(
	cost_sum_type *dest,
	const cost_type *buffer_in,
	int width,
	int height,
	int num_paths,
	cudaStream_t stream);

template void sum_energy_all_paths<128u>(
	cost_sum_type *dest,
	const cost_type *buffer_in,
	int width,
	int height,
	int num_paths,
	cudaStream_t stream);

template void sum_energy_all_paths<256u>(
	cost_sum_type *dest,
	const cost_type *buffer_in,
	int width,
	int height,
	int num_paths,
	cudaStream_t stream);

}

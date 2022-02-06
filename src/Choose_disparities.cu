#include "Winner_Takes_All.hpp"
#include <iterator>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
// #include <thrust/sort.h>

using namespace thrust::placeholders;

namespace sgm {

__global__ void rotate_3D_x_kernel(cost_sum_type *dest,
                                    const cost_sum_type *cost_in, 
                                    int width, int height,
                                    int min_disparity,
                                    int max_disparity)
{
    const unsigned int width_height = width * height;
    const unsigned int width_disparity = width * (max_disparity - min_disparity);
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y;
    if (x > width -1 || y > height - 1) return;
    const unsigned int d = blockIdx.z;
    dest[x + d * width + y * width_disparity] = cost_in[x + y * width + (d + min_disparity) * width_height];
}

void choose_disparities(
	output_type *dest,
	const cost_sum_type *cost_in,
	int width,
	int height,
	int min_disparity,
	int max_disparity,
	cudaStream_t stream)
{
	printf("My choose_disparities\n");
    const size_t buffer_step = height * width * (max_disparity - min_disparity + 1);
    DeviceBuffer<cost_sum_type> cost_transposed = DeviceBuffer<cost_sum_type>(buffer_step);
    
    int num_threads = 1024;
	int block_y_dim = nextPowerOf2(height); // 512
	int block_x_dim = num_threads / block_y_dim; // 2
	int grid_dim = width / block_x_dim; // 238
	if (grid_dim * block_x_dim < width) {
		grid_dim += 1;
	}
	const dim3 bdim(block_x_dim, block_y_dim, 1);
	const dim3 gdim(grid_dim, 1, max_disparity - min_disparity);
    rotate_3D_x_kernel<<<gdim, bdim, 0, stream>>>(cost_transposed.mutable_data(),
	    cost_in, width, height, min_disparity, max_disparity);
    
    const int col = width;
    const int row = max_disparity - min_disparity;
    thrust::device_vector<cost_sum_type> minval(col * height);
    thrust::device_vector<int> minidx(col * height);
    thrust::device_vector<cost_sum_type> ccccc(
        cost_transposed.data() + min_disparity * col, 
        cost_transposed.data() + min_disparity * col + row * col * height);
    for (int y0 = 0 ; y0 < height; ++y0) {
        thrust::reduce_by_key(
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        _1 / row),
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        _1 / row) + row * col,
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                thrust::make_permutation_iterator(
                                        ccccc.begin() + row * col * y0,
                                        thrust::make_transform_iterator(
                                                thrust::make_counting_iterator(0),
                                                (_1 % row) * col + _1 / row) ),
                                thrust::make_transform_iterator(
                                        thrust::make_counting_iterator(0),
                                        _1 % row))),
                thrust::make_discard_iterator(),
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                minval.begin() + col * y0,
                                minidx.begin() + col * y0) ),
                thrust::equal_to<int>(),
                thrust::minimum<thrust::tuple<cost_sum_type, int> >()
        );
        // std::copy(minidx.begin(), minidx.end(), std::ostream_iterator<int>(std::cout, "-"));
        // std::cout << std::endl;
    }
    thrust::device_vector<output_type> minidx_d(minidx.begin(), minidx.end());
    void* src_void_ptr = thrust::raw_pointer_cast(minidx_d.data());
    void* dest_void_ptr = dest;
    cudaMemcpy(dest_void_ptr, src_void_ptr, col * height * sizeof(output_type), cudaMemcpyDeviceToDevice);
    cost_transposed.destroy();
#ifdef DEBUG
    output_type lookup[height * width];
    printf("dest.size()=%zd\n", sizeof(lookup)/sizeof(lookup[0]));
    cudaMemcpy(lookup, dest, sizeof(output_type) * height * width, cudaMemcpyDeviceToHost);
    for (int y = 0; y < height; ++y) {
        printf("y:%3.3d", y);
        for (int x = 0; x < width; ++x) {
            output_type val = lookup[x + y * width];
            printf("%4.1d", val);
        }
        printf("\n");
    }
#endif
	printf("My choose_disparities end\n");
}

} // namespace sgm
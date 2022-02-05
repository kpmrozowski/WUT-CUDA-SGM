#include "WinnerTakesAll.hpp"
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

__global__ void choose_disparities_kernel(output_type *dest,
                                          const cost_sum_type *cost_in,
                                          int width, int height,
                                          int max_disparity)
{
  ;
}

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
    dest[x + d * width + (height - 1 - y) * width_disparity] = cost_in[x + y * width + (d + min_disparity) * width_height];
}

__global__ void rotate_3D_y_kernel(cost_sum_type *dest,
                                    const cost_sum_type *cost_in, 
                                    int width, int height,
                                    int min_disparity,
                                    int max_disparity)
{
    const unsigned int disparity = max_disparity - min_disparity;
    const unsigned int width_height = width * height;
    // const unsigned int width_disparity = width * (max_disparity - min_disparity);
    const unsigned int height_disparity = height * (max_disparity - min_disparity);
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y;
    if (x > width -1 || y > height - 1) return;
    const unsigned int d = blockIdx.z;
    dest[d + y * disparity + (width - 1 - x) * height_disparity] = cost_in[x + y * width + (d + min_disparity) * width_height];
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
    for (int y0 = 0 ; y0 < height; ++y0) {
        thrust::device_vector<cost_sum_type> ccccc(
            cost_transposed.data() + row * col * (y0 + 0) + min_disparity * col, 
            cost_transposed.data() + row * col * (y0 + 1) + min_disparity * col);

        thrust::device_vector<cost_sum_type> minval(col);
        thrust::device_vector<int> minidx(col);

        thrust::reduce_by_key(
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator((int) 0),
                        _1 / row),
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator((int) 0),
                        _1 / row) + row * col,
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                thrust::make_permutation_iterator(
                                        ccccc.begin(),
                                        thrust::make_transform_iterator(
                                                thrust::make_counting_iterator((int) 0), (_1 % row) * col + _1 / row)),
                                thrust::make_transform_iterator(
                                        thrust::make_counting_iterator((int) 0), _1 % row))),
                thrust::make_discard_iterator(),
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                minval.begin(),
                                minidx.begin())),
                thrust::equal_to<int>(),
                thrust::minimum<thrust::tuple<cost_sum_type, int> >()
        );
        thrust::device_vector<output_type> minidx_d(minidx.begin(), minidx.end());
        void* src_void_ptr = thrust::raw_pointer_cast(minidx_d.data());
        void* dest_void_ptr = dest + col * y0;
        cudaMemcpy(dest_void_ptr, src_void_ptr, col * sizeof(output_type), cudaMemcpyDeviceToDevice);
        // std::copy(minidx.begin(), minidx.end(), std::ostream_iterator<int>(std::cout, "-"));
        std::cout << std::endl;
    }
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




// template <typename Iterator>
// class strided_range
// {
//     public:

//     typedef typename thrust::iterator_difference<Iterator>::type difference_type;

//     struct stride_functor : public thrust::unary_function<difference_type,difference_type>
//     {
//         difference_type stride;

//         stride_functor(difference_type stride)
//             : stride(stride) {}

//         __host__ __device__
//         difference_type operator()(const difference_type& i) const
//         { 
//             return stride * i;
//         }
//     };

//     typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
//     typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
//     typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

//     // type of the strided_range iterator
//     typedef PermutationIterator iterator;

//     // construct strided_range for the range [first,last)
//     strided_range(Iterator first, Iterator last, difference_type stride)
//         : first(first), last(last), stride(stride) {}

//     iterator begin(void) const
//     {
//         return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
//     }

//     iterator end(void) const
//     {
//         return begin() + ((last - first) + (stride - 1)) / stride;
//     }

//     protected:
//     Iterator first;
//     Iterator last;
//     difference_type stride;
// };


// /**************************************************************/
// /* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
// /**************************************************************/
// template< typename T >
// struct mod_functor {
//     __host__ __device__ T operator()(T a, T b) { return a % b; }
// };

void find2largest(
	output_type *dest,
	const cost_sum_type *cost_in,
	int width,
	int height,
	int min_disparity,
	int max_disparity,
	cudaStream_t stream)
{
	// printf("My find2largest\n");

    // /***********************/
    // /* SETTING THE PROBLEM */
    // /***********************/
    // const size_t buffer_step = height * width * (max_disparity - min_disparity);
    // DeviceBuffer<cost_sum_type> cost_transposed = DeviceBuffer<cost_sum_type>(buffer_step);
    
    // int num_threads = 1024;
	// int block_y_dim = nextPowerOf2(height); // 512
	// int block_x_dim = num_threads / block_y_dim; // 2
	// int grid_dim = width / block_x_dim; // 238
	// if (grid_dim * block_x_dim < width) {
	// 	grid_dim += 1;
	// }
	// const dim3 bdim(block_x_dim, block_y_dim, 1);
	// const dim3 gdim(grid_dim, 1, max_disparity - min_disparity);
    // rotate_3D_y_kernel<<<gdim, bdim, 0, stream>>>(cost_transposed.mutable_data(),
	//     cost_in, width, height, min_disparity, max_disparity);
    // const int Ncols = max_disparity - min_disparity;
    // const int Nrows = height;
    // const int x0 = width / 2; // height
    // // float initc[] = { 
    // //         0, 10, 20, 3, 40, 
    // //         1,  2, 30, 5, 10 };
    // thrust::device_vector<cost_sum_type> d_matrix(
    //     cost_transposed.data() + Nrows * Ncols * (x0 + 0) + min_disparity * Ncols, 
    //     cost_transposed.data() + Nrows * Ncols * (x0 + 1) + min_disparity * Ncols);

    // // const int Nrows = 4;
    // // const int Ncols = 6;

    // // // --- Random uniform integer distribution between 10 and 99
    // // thrust::default_random_engine rng;
    // // thrust::uniform_int_distribution<int> dist(10, 99);

    // // // --- Matrix allocation and initialization
    // // thrust::device_vector<float> d_matrix(Nrows * Ncols);
    // // for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist(rng);

    // for(int i = 0; i < Nrows; i++) {
    //     std::cout << "[ ";
    //     for(int j = 0; j < Ncols; j++)
    //         std::cout << d_matrix[i * Ncols + j] << " ";
    //     std::cout << "]\n";
    // }

    // /******************/
    // /* APPROACH NR. 2 */
    // /******************/
    // // --- Computing row indices vector
    // thrust::device_vector<int> d_row_indices(Nrows * Ncols);
    // thrust::transform(
    //         thrust::make_counting_iterator(0), 
    //         thrust::make_counting_iterator(Nrows * Ncols), 
    //         thrust::make_constant_iterator(Ncols), 
    //         d_row_indices.begin(), 
    //         thrust::divides<int>()
    // );

    // // --- Computing column indices vector
    // thrust::device_vector<int> d_column_indices(Nrows * Ncols);
    // thrust::transform(
    //         thrust::make_counting_iterator(0), 
    //         thrust::make_counting_iterator(Nrows * Ncols), 
    //         thrust::make_constant_iterator(Ncols), 
    //         d_column_indices.begin(), 
    //         mod_functor<int>()
    // );

    // // --- int and float iterators
    // typedef thrust::device_vector<int>::iterator            IntIterator;
    // typedef thrust::device_vector<cost_sum_type>::iterator  CostIterator;

    // // --- Relevant tuples of int and float iterators
    // typedef thrust::tuple<IntIterator, IntIterator>         IteratorTuple1;
    // typedef thrust::tuple<CostIterator, IntIterator>        IteratorTuple2;

    // // --- zip_iterator of the relevant tuples
    // typedef thrust::zip_iterator<IteratorTuple1>            ZipIterator1;
    // typedef thrust::zip_iterator<IteratorTuple2>            ZipIterator2;

    // // --- zip_iterator creation
    // ZipIterator1 iter1(
    //         thrust::make_tuple(
    //                 d_row_indices.begin(), 
    //                 d_column_indices.begin()
    //         )
    // );

    // thrust::stable_sort_by_key(
    //         d_matrix.begin(), 
    //         d_matrix.end(), 
    //         iter1
    // );

    // ZipIterator2 iter2(
    //         thrust::make_tuple(
    //                 d_matrix.begin(), 
    //                 d_column_indices.begin()
    //         )
    // );

    // thrust::stable_sort_by_key(
    //         d_row_indices.begin(), 
    //         d_row_indices.end(), 
    //         iter2
    // );

    // typedef thrust::device_vector<int>::iterator Iterator;

    // // --- Strided access to the sorted array
    // strided_range<Iterator> d_min_indices_1(
    //         d_column_indices.begin(), 
    //         d_column_indices.end(), 
    //         Ncols);
    // strided_range<Iterator> d_min_indices_2(
    //         d_column_indices.begin() + 1, 
    //         d_column_indices.end() + 1, 
    //         Ncols
    // );

    // printf("\n\n");
    // for(int i = 0; i < Nrows; i++) {
    //     std::cout << "[ ";
    //     for(int j = 0; j < Ncols; j++)
    //         std::cout << d_matrix[i * Ncols + j] << " ";
    //     std::cout << "]\n";
    // }

    // // printf("\n\n");
    // // for (auto it = d_min_indices_1.begin(); it != d_min_indices_1.end(); ++it)
    // //     printf("%3.0d-", *it);
    // std::copy(d_min_indices_1.begin(), d_min_indices_1.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    // // std::copy(d_min_indices_2.begin(), d_min_indices_2.end(), std::ostream_iterator<int>(std::cout, " "));
    // // std::cout << std::endl;
    
    // printf("My find2largest end\n");
}

} // namespace sgm
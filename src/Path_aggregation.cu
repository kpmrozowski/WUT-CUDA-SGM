#include "Path_aggregation.hpp"

namespace sgm {


template <size_t MAX_DISPARITY>
PathAggregation<MAX_DISPARITY>::PathAggregation()
	: m_path_cost_buffer()
{  }


template <size_t MAX_DISPARITY>
void PathAggregation<MAX_DISPARITY>::enqueue(
	const feature_type *left,
	const feature_type *right,
	int width,
	int height,
	PathType path_type,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const unsigned int num_paths = path_type == PathType::SCAN_4PATH ? 4 : 8;

	const size_t buffer_size = width * height * MAX_DISPARITY * num_paths;
	if(m_path_cost_buffer.size() != buffer_size){
		m_path_cost_buffer = DeviceBuffer<cost_type>(buffer_size);
	}
	const size_t buffer_step = width * height * MAX_DISPARITY;
	cudaStreamSynchronize(stream);

	// path_aggregation::enqueue_aggregate_up2down_path<MAX_DISPARITY>(
	// 	m_path_cost_buffer.data() + 0 * buffer_step,
	// 	left, right, width, height, p1, p2, min_disp, m_streams[0]);
	// path_aggregation::enqueue_aggregate_down2up_path<MAX_DISPARITY>(
	// 	m_path_cost_buffer.data() + 1 * buffer_step,
	// 	left, right, width, height, p1, p2, min_disp, m_streams[1]);
	// path_aggregation::enqueue_aggregate_left2right_path<MAX_DISPARITY>(
	// 	m_path_cost_buffer.data() + 2 * buffer_step,
	// 	left, right, width, height, p1, p2, min_disp, m_streams[2]);
	// path_aggregation::enqueue_aggregate_right2left_path<MAX_DISPARITY>(
	// 	m_path_cost_buffer.data() + 3 * buffer_step,
	// 	left, right, width, height, p1, p2, min_disp, m_streams[3]);

	// if (path_type == PathType::SCAN_8PATH) {
	// 	path_aggregation::enqueue_aggregate_upleft2downright_path<MAX_DISPARITY>(
	// 		m_path_cost_buffer.data() + 4 * buffer_step,
	// 		left, right, width, height, p1, p2, min_disp, m_streams[4]);
	// 	path_aggregation::enqueue_aggregate_upright2downleft_path<MAX_DISPARITY>(
	// 		m_path_cost_buffer.data() + 5 * buffer_step,
	// 		left, right, width, height, p1, p2, min_disp, m_streams[5]);
	// 	path_aggregation::enqueue_aggregate_downright2upleft_path<MAX_DISPARITY>(
	// 		m_path_cost_buffer.data() + 6 * buffer_step,
	// 		left, right, width, height, p1, p2, min_disp, m_streams[6]);
	// 	path_aggregation::enqueue_aggregate_downleft2upright_path<MAX_DISPARITY>(
	// 		m_path_cost_buffer.data() + 7 * buffer_step,
	// 		left, right, width, height, p1, p2, min_disp, m_streams[7]);
	// }
}
template class PathAggregation< 64>;
template class PathAggregation<128>;
template class PathAggregation<256>;

}
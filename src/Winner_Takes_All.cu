#include "Winner_Takes_All.hpp"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace sgm {

WinnerTakesAll::WinnerTakesAll()
	: m_disparities()
{
}

void WinnerTakesAll::compute(
	const cost_sum_type *cost_in,
	int width,
	int height,
	int min_disparity,
	int max_disparity,
	cudaStream_t stream)
{
	const size_t buffer_step = width * height;
	if(m_disparities.size() != buffer_step){
		m_disparities = DeviceBuffer<cost_sum_type>(buffer_step);
	}
	choose_disparities(
		m_disparities.mutable_data(), cost_in, width, height, min_disparity,
		max_disparity, stream);
}

}
#include <cstdio>
#include "Matching_cost.hpp"

namespace sgm {

template <size_t MAX_DISPARITY>
MatchingCost<MAX_DISPARITY>::MatchingCost()
	: m_cost_cube()
{ }

template <size_t MAX_DISPARITY>
void enqueue_matching_cost(
	feature_type *dest,
	const feature_type *ctL,
	const feature_type *ctR,
	int width,
	int height)
{
	printf("My matching cost\n");
}

template <size_t MAX_DISPARITY>
void MatchingCost<MAX_DISPARITY>::enqueue(
	const feature_type *ctL,
	const feature_type *ctR,
	int width,
	int height)
{
	if(m_cost_cube.size() != static_cast<size_t>(width * height)){
		m_cost_cube = DeviceBuffer<feature_type>(width * height * MAX_DISPARITY);
	}
	enqueue_matching_cost<MAX_DISPARITY>(
		m_cost_cube.data(), ctL, ctR, width, height);
}

template class MatchingCost< 64>;
template class MatchingCost<128>;
template class MatchingCost<256>;

}
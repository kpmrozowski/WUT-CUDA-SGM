#include "Engine_SGM.hpp"
#include "stdio.h"
#include "Census_transform.hpp"
#include "Matching_cost.hpp"
#include "Energy.hpp"

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class Engine_SGM<T, MAX_DISPARITY>::Impl {

private:
	CensusTransform<T> m_census_left;
	CensusTransform<T> m_census_right;
	MatchingCost<MAX_DISPARITY> m_matching_cost;
	EnergyAgregator<MAX_DISPARITY> m_energy_agregator;

public:
	Impl()
		: m_census_left()
		, m_census_right()
		, m_matching_cost()
	{ }

	void compute() {}
	void compute(
		output_type *dest_left,
		output_type *dest_right,
		const input_type *src_left,
		const input_type *src_right,
		int width,
		int height,
		const Parameters& param,
		cudaStream_t stream)
	{
		printf("Stereo starts\n");
		m_census_left.compute(src_left, width, height);
		m_census_right.compute(src_right, width, height);
		m_matching_cost.compute(
			m_census_left.get_output(), 
			m_census_right.get_output(), 
			width, height);
		m_energy_agregator.compute(
			m_matching_cost.get_output(),
			width, height, param.num_paths,
			param.P1, param.P2, param.min_disp, stream);
		printf("Stereo ends\n");
	}
};


template <typename T, size_t MAX_DISPARITY>
Engine_SGM<T, MAX_DISPARITY>::Engine_SGM()
	: m_impl(new Impl())
{ }

template <typename T, size_t MAX_DISPARITY>
Engine_SGM<T, MAX_DISPARITY>::~Engine_SGM() = default;


template <typename T, size_t MAX_DISPARITY>
void Engine_SGM<T, MAX_DISPARITY>::execute()
{
	m_impl->compute();
}

template <typename T, size_t MAX_DISPARITY>
void Engine_SGM<T, MAX_DISPARITY>::execute(
	output_type *dest_left,
	output_type *dest_right,
	const input_type *src_left,
	const input_type *src_right,
	int width,
	int height,
	const Parameters& param)
{
	m_impl->compute(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		param, 0);
	cudaStreamSynchronize(0);
}


template class Engine_SGM<uint8_t,   64>;
template class Engine_SGM<uint8_t,  128>;
template class Engine_SGM<uint8_t,  256>;
template class Engine_SGM<uint16_t,  64>;
template class Engine_SGM<uint16_t, 128>;
template class Engine_SGM<uint16_t, 256>;

}

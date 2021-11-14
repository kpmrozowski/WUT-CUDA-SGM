#include "Engine_SGM.hpp"
#include "stdio.h"
#include "census_transform.hpp"
// #include "path_aggregation.hpp"
// #include "winner_takes_all.hpp"

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class Engine_SGM<T, MAX_DISPARITY>::Impl {

private:
	CensusTransform<T> m_census_left;
	CensusTransform<T> m_census_right;

public:
	Impl()
		: m_census_left()
		, m_census_right()
	{ }

	void enqueue() {}
	void enqueue(
		output_type *dest_left,
		output_type *dest_right,
		const input_type *src_left,
		const input_type *src_right,
		int width,
		int height,
		int src_pitch,
		int dst_pitch,
		const Parameters& param,
		cudaStream_t stream)
	{
		printf("Stereo starts\n");
		m_census_left.enqueue(src_left, width, height, src_pitch, stream);
		m_census_right.enqueue(src_right, width, height, src_pitch, stream);
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
	m_impl->enqueue();
	cudaStreamSynchronize(0);
}

template <typename T, size_t MAX_DISPARITY>
void Engine_SGM<T, MAX_DISPARITY>::execute(
	output_type *dest_left,
	output_type *dest_right,
	const input_type *src_left,
	const input_type *src_right,
	int width,
	int height,
	int src_pitch,
	int dst_pitch,
	const Parameters& param)
{
	m_impl->enqueue(
		dest_left, dest_right,
		src_left, src_right,
		width, height,
		src_pitch, dst_pitch,
		param,
		0);
	cudaStreamSynchronize(0);
}


template class Engine_SGM<uint8_t,   64>;
template class Engine_SGM<uint8_t,  128>;
template class Engine_SGM<uint8_t,  256>;
template class Engine_SGM<uint16_t,  64>;
template class Engine_SGM<uint16_t, 128>;
template class Engine_SGM<uint16_t, 256>;

}

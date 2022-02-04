#include "Energy.hpp"

namespace sgm {

template <size_t MAX_DISPARITY>
EnergyAgregator<MAX_DISPARITY>::EnergyAgregator()
	: m_energy_buffer()
{
	for(unsigned int i = 0; i < MAX_NUM_PATHS; ++i){
		cudaStreamCreate(&m_streams[i]);
		cudaEventCreate(&m_events[i]);
	}
}

template <size_t MAX_DISPARITY>
EnergyAgregator<MAX_DISPARITY>::~EnergyAgregator(){
	for(unsigned int i = 0; i < MAX_NUM_PATHS; ++i){
		cudaStreamSynchronize(m_streams[i]);
		cudaStreamDestroy(m_streams[i]);
		cudaEventDestroy(m_events[i]);
	}
}


template <size_t MAX_DISPARITY>
void EnergyAgregator<MAX_DISPARITY>::compute(
	const cost_type *cost_in,
	int width,
	int height,
	int num_paths,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream)
{
	const size_t buffers_size = width * height * MAX_DISPARITY * num_paths;
	const size_t buffer_step = width * height * MAX_DISPARITY;
	if(m_energy_buffer.size() != buffer_step){
		m_energy_buffer = DeviceBuffer<cost_sum_type>(buffer_step);
	}
	DeviceBuffer<cost_type> steps_buffer = DeviceBuffer<cost_type>(buffers_size);
	cudaStreamSynchronize(stream);

	switch (num_paths) {
		// oblique
		case 8:
			compute_energy_downL2upR<MAX_DISPARITY>(
				steps_buffer.data() + 7 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[7]);
		case 7:
			compute_energy_downR2upL<MAX_DISPARITY>(
				steps_buffer.data() + 6 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[6]);
		case 6:
			compute_energy_upR2downL<MAX_DISPARITY>(
				steps_buffer.data() + 5 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[5]);
		case 5:
			compute_energy_upL2downR<MAX_DISPARITY>(
				steps_buffer.data() + 4 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[4]);
		// horizontal
		case 4:
			compute_energy_R2L<MAX_DISPARITY>(
				steps_buffer.data() + 3 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[3]);
		case 3:
			compute_energy_L2R<MAX_DISPARITY>(
				steps_buffer.data() + 2 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[2]);
		// vertical
		case 2:
			compute_energy_down2up<MAX_DISPARITY>(
				steps_buffer.data() + 1 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[1]);
		case 1:
			compute_energy_up2down<MAX_DISPARITY>(
				steps_buffer.data() + 0 * buffer_step,
				cost_in, width, height, p1, p2, min_disp, m_streams[0]);
			break;
		default:
			printf("Incorrect number of paths, should be > 0 and < 9");
			return;
	}
	sum_energy_all_paths<MAX_DISPARITY>(
		m_energy_buffer.data(), steps_buffer.data(),
		width, height, num_paths, m_streams[0]);
}

template class EnergyAgregator< 64>;
template class EnergyAgregator<128>;
template class EnergyAgregator<256>;

}
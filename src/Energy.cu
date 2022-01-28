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
	const size_t buffer_size = width * height * MAX_DISPARITY * num_paths;
	if(m_energy_buffer.size() != buffer_size){
		m_energy_buffer = DeviceBuffer<cost_type>(buffer_size);
	}
	const size_t buffer_step = width * height * MAX_DISPARITY;
	cudaStreamSynchronize(stream);

	// vertical
	if (num_paths < 1 || num_paths > 8) return;
	compute_energy_up2down<MAX_DISPARITY>(
		m_energy_buffer.data() + 0 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[0]);
	if (num_paths < 2) return;
	compute_energy_down2up<MAX_DISPARITY>(
		m_energy_buffer.data() + 1 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[1]);
	// horizontal
	if (num_paths < 3) return;
	compute_energy_L2R<MAX_DISPARITY>(
		m_energy_buffer.data() + 2 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[2]);
	if (num_paths < 4) return;
	compute_energy_R2L<MAX_DISPARITY>(
		m_energy_buffer.data() + 3 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[3]);
	// oblique
	if (num_paths < 5) return;
	compute_energy_upL2downR<MAX_DISPARITY>(
		m_energy_buffer.data() + 4 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[4]);
	if (num_paths < 6) return;
	compute_energy_upR2downL<MAX_DISPARITY>(
		m_energy_buffer.data() + 5 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[5]);
	if (num_paths < 7) return;
	compute_energy_downR2upL<MAX_DISPARITY>(
		m_energy_buffer.data() + 6 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[6]);
	if (num_paths < 8) return;
	compute_energy_downL2upR<MAX_DISPARITY>(
		m_energy_buffer.data() + 7 * buffer_step,
		cost_in, width, height, p1, p2, min_disp, m_streams[7]);
}
template class EnergyAgregator< 64>;
template class EnergyAgregator<128>;
template class EnergyAgregator<256>;

}
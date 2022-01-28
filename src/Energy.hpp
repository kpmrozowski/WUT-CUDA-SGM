#ifndef SGM_ENERGY_HPP
#define SGM_ENERGY_HPP

#include <Device_buffer.hpp>
#include <Parameters.hpp>
#include <types.hpp>

namespace sgm {

static constexpr unsigned int WARP_SIZE = 32u;

template <size_t MAX_DISPARITY>
class EnergyAgregator {

private:
	static const unsigned int MAX_NUM_PATHS = 8;

	DeviceBuffer<cost_type> m_energy_buffer;
	cudaStream_t m_streams[MAX_NUM_PATHS];
	cudaEvent_t m_events[MAX_NUM_PATHS];
	
public:
	EnergyAgregator();
	~EnergyAgregator();

	const cost_type *get_output() const {
		return m_energy_buffer.data();
	}

	void compute(
		const cost_type *cost_in,
		int width,
		int height,
		int num_paths,
		unsigned int p1,
		unsigned int p2,
		int min_disp,
		cudaStream_t stream);

};

// vertical
template <unsigned int MAX_DISPARITY>
void compute_energy_up2down(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void compute_energy_down2up(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

// horizontal
template <unsigned int MAX_DISPARITY>
void compute_energy_L2R(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void compute_energy_R2L(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

// oblique
template <unsigned int MAX_DISPARITY>
void compute_energy_upL2downR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void compute_energy_upR2downL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void compute_energy_downR2upL(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);

template <unsigned int MAX_DISPARITY>
void compute_energy_downL2upR(
	cost_type *dest,
	const cost_type *cost_in,
	int width,
	int height,
	unsigned int p1,
	unsigned int p2,
	int min_disp,
	cudaStream_t stream);
}

#endif // SGM_ENERGY_HPP

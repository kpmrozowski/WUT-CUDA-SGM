#ifndef SGM_ENGINE_HPP
#define SGM_ENGINE_HPP
#include <memory>
#include <cstdint>
#include <types.hpp>
#include <Parameters.hpp>

namespace sgm {

template <typename T, size_t MAX_DISPARITY>
class Engine_SGM {

public:
	using input_type = T;
	using output_type = sgm::output_type;

private:
	class Impl;
	std::unique_ptr<Impl> m_impl;

public:
	Engine_SGM();
	~Engine_SGM();

	void execute();
	void execute(
		output_type *dest_left,
		const input_type *src_left,
		const input_type *src_right,
		int width,
		int height,
		const Parameters& param);

};

}

#endif // SGM_ENGINE_HPP

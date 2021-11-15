#ifndef SGM_IENGINE_HPP
#define SGM_IENGINE_HPP

#include "Engine_SGM.hpp"

namespace sgm {

class IEngine_SGM {
public:
	using output_type = sgm::output_type;
	virtual void execute() = 0;
	virtual void execute(output_type* dst_L, output_type* dst_R, const void* src_L, const void* src_R, 
		int w, int h, Parameters& param) = 0;
	
	virtual ~IEngine_SGM() {}
};

template <typename input_type, int MAX_DISPARITY>
class Engine_SGM_Impl : public IEngine_SGM {
public:
	void execute() override
	{
		sgm_engine_.execute();
	}
	void execute(output_type* dst_L, output_type* dst_R, const void* src_L, const void* src_R,
		int w, int h, Parameters& param) override
	{
		sgm_engine_.execute(dst_L, dst_R, (const input_type*)src_L, (const input_type*)src_R, w, h, param);
	}
private:
	Engine_SGM<input_type, MAX_DISPARITY> sgm_engine_;
};

}

#endif // SGM_IENGINE_HPP

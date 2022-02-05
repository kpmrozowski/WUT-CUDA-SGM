#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../src/Stereo_SGM.hpp"

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

int main()
{
	// cv::Mat  left = cv::imread("../../data/cone/im2.png", CV_16U);
	// cv::Mat right = cv::imread("../../data/cone/im6.png", CV_16U);
	cv::Mat  left = cv::imread("im2.png", CV_16U);
	cv::Mat right = cv::imread("im6.png", CV_16U);

	const int disp_size = 64;
	const int P1 = 20;
	const int P2 = 64;
	const int num_paths = 8;
	const int min_disp = 3;
	const int LR_max_diff = 1;

	ASSERT_MSG(!left.empty() && !right.empty(), "imread failed.");
	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	printf("type=%d\n", left.type());
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");

	const int depth = left.type() == CV_8U ? 8 : 16;

	const sgm::Parameters param(P1, P2, false, num_paths, min_disp, LR_max_diff);
	sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, depth, param);

	cv::Mat disparity(left.size(), CV_16S);

	ssgm.execute(disparity.data, left.data, right.data);

	// // create mask for invalid disp
	// cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	// // show image
	// cv::Mat disparity_8u, disparity_color;
	// disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
	// cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	// disparity_8u.setTo(0, mask);
	// disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
	// if (left.type() != CV_8U)
	// 	cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8U);

	// std::vector<cv::Mat> images = { disparity_8u, disparity_color, left };
	// std::vector<std::string> titles = { "disparity", "disparity color", "input" };

	// std::cout << "Hot keys:" << std::endl;
	// std::cout << "\tESC - quit the program" << std::endl;
	// std::cout << "\ts - switch display (disparity | colored disparity | input image)" << std::endl;

	// int mode = 0;
	// while (true) {

	// 	cv::setWindowTitle("image", titles[mode]);
	// 	cv::imshow("image", images[mode]);

	// 	const char c = cv::waitKey(0);
	// 	if (c == 's')
	// 		mode = (mode < 2 ? mode + 1 : 0);
	// 	if (c == 27)
	// 		break;
	// }

	return 0;
}

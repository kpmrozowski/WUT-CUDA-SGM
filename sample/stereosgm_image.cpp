#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../src/Stereo_SGM.hpp"

#define ASSERT_MSG(expr, msg)     \
  if (!(expr)) {                  \
    std::cerr << msg << std::endl;\
    std::exit(EXIT_FAILURE);      \
  }

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv,
		"{@left_img   | data/cone/im2.png | path to input left image                                         }"
		"{@right_img  | data/cone/im6.png | path to input right image                                        }"
		"{min_disp    |      3 | minimum disparity value                                                     }"
		"{max_disp    |     64 | maximum disparity value                                                     }"
		"{P1          |      5 | penalty on the disparity change by plus or minus 1 between nieghbor pixels  }"
		"{P2          |     20 | penalty on the disparity change by more than 1 between neighbor pixels      }"
		"{num_paths   |      8 | number of scanlines used in cost aggregation                                }"
		"{help h      |        | display this help and exit                                                  }");
    
    if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
    
    cv::Mat  left = cv::imread(parser.get<cv::String>("@left_img"), CV_16U);
	cv::Mat right = cv::imread(parser.get<cv::String>("@right_img"), CV_16U);
	// cv::Mat  left = cv::imread("../../data/cone/im2.png", CV_16U);
	// cv::Mat right = cv::imread("../../data/cone/im6.png", CV_16U);
    
    const int min_disp = parser.get<int>("min_disp");
	const int max_disp = parser.get<int>("max_disp");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const int num_paths = parser.get<int>("num_paths");
    
    ASSERT_MSG(!left.empty() && !right.empty(), "imread failed.");
	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(max_disp == 64 || max_disp == 128 || max_disp == 256, "disparity size must be 64, 128 or 256.");
    
    const int depth = left.type() == CV_8U ? 8 : 16;
    const sgm::Parameters param(P1, P2, num_paths, min_disp);
    cv::Mat disparity(left.size(), CV_16S);
    sgm::StereoSGM ssgm(left.cols, left.rows, max_disp, depth, param);
	ssgm.execute(disparity.data, left.data, right.data);
    
    // show image
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255. / max_disp);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	if (left.type() != CV_8U)
		cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    std::vector<cv::Mat> images = { disparity_8u, disparity_color, left };
	std::vector<std::string> titles = { "disparity gpu", "disparity gpu color", "input" };
    
    std::cout << "Hot keys:" << std::endl;
	std::cout << "\tESC - quit the program" << std::endl;
	std::cout << "\ts - switch display (disparity | colored disparity | input image)" << std::endl;
    
    int mode = 0;
	while (true) {
		cv::setWindowTitle("image", titles[mode]);
		cv::imshow("image", images[mode]);
		const char c = cv::waitKey(0);
		if (c == 's')
			mode = (mode < 2 ? mode + 1 : 0);
		if (c == 27)
			break;
	}

	return 0;
}

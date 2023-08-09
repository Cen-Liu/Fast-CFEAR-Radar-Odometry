#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <bitset>
#include <opencv2/core/core.hpp>
#include "Functions.h"

using namespace std;
using namespace cv;

// K-strongest filtering
std::vector<RadarPoint> KStrong(cv::Mat img_polar_oxford, size_t K, double Zmin, double Dmin, double Dmax, double range_res, double range_down) {

	// Delete first 11 columns (formats) of the Oxford radar image
	cv::Mat img_polar_nonform = FormatDelete(img_polar_oxford);

	// Downsample the range (columns) of polar images
	if (range_down != 1.0)
		cv::resize(img_polar_nonform, img_polar_nonform, Size(), range_down, 1, INTER_AREA);

	// Set range to be considered
	cv::Mat img_polar = RangeSelect(img_polar_nonform, Dmin, Dmax, range_res);

	// K-strongest filtering
	const int M = img_polar.rows;
	const int N = img_polar.cols;
	double Kmax = 0;

	std::vector<RadarPoint> kstrong_pointset;

	for (int i = 0; i < M; i++) {
		double timestamp = FormatInfo(img_polar_oxford.row(i).colRange(0, 8));
		double sweep_counter = FormatInfo(img_polar_oxford.row(i).colRange(8, 10));
		double azimuth = sweep_counter / 2800.0 * M_PI;

		std::vector<double> Val;

		// Pixel values of each row
		for (int j = 0; j < N; j++) {
			double value = img_polar.at<double>(i, j);
			Val.push_back(value);
		}

		// The K-th maximum pixel value
		sort(Val.rbegin(), Val.rend());
		Kmax = Val[K - 1];

		// Preserve K-strongest pixels from the central of the image
		int count = 0;
		for (int j = 0; j < N; j++) {
			double value = img_polar.at<double>(i, j);

			if (value >= Kmax && value >= Zmin) {
				double range = (static_cast<double>(j) + 1.0) * range_res;

				RadarPoint kstrong_point;
				kstrong_point.timestamp = timestamp;
				kstrong_point.azimuth = azimuth;
				kstrong_point.range = range;
				kstrong_point.coordinate(0) = range * cos(azimuth);
				kstrong_point.coordinate(1) = range * sin(azimuth);
				kstrong_point.coordinate(2) = 0.0;
				kstrong_pointset.push_back(kstrong_point);

				count++;
			}
			if (count >= K)
				break;
		}
	}

	return kstrong_pointset;
}


// Delete the first 11 columns (formats) of Oxford radar image
cv::Mat FormatDelete(cv::Mat img_in) {

	const int format_num = 11;
	cv::Mat img_out = img_in.colRange(format_num, img_in.cols).clone();

	return img_out;
}


// Select range of polar image
cv::Mat RangeSelect(cv::Mat img_in, double Dmin, double Dmax, double range_res) {

	const int M = img_in.rows;
	const int col_bgn = static_cast<int>(floor(Dmin / range_res));
	const int col_end = static_cast<int>(floor(Dmax / range_res));

	// Set maximun range
	cv::Mat img_out = img_in.colRange(0, col_end).clone();

	// Set pixels that below minimum range to 0
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < col_bgn; j++) {
			img_out.at<double>(i, j) = 0.0;
		}
	}

	return img_out;
}


// Extract format information from a polar radar beam
double FormatInfo(cv::Mat img_in) {

	const int N = img_in.cols;
	const int M = img_in.rows;

	std::string str_info;
	for (int i = 0; i < N; i++) {
		int val = static_cast<int>(img_in.at<double>(0, i));
		std::string str = std::bitset<8>(val).to_string();
		str_info = str + str_info;
	}
	double info = static_cast<double>(std::stoll(str_info, nullptr, 2));

	return info;
}
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;

// A point in polar radar image
struct RadarPoint {
	double timestamp;
	double azimuth;
	double range;
	Eigen::Vector3d coordinate; // Coordinate in Cartesian image frame; Origin: center of Cartesian image; X-axis: rightwards, Y-axis: downwards
};

// A scan includes radar points
struct Scan {
	Eigen::Vector3d timespan; // Begin, middle and end time of a scan
	std::vector<double> timestamps;
	std::vector<double> azimuths;
	std::vector<double> ranges;
	std::vector<Eigen::Vector3d> means; // Coordinate in Cartesian image frame
	std::vector<Eigen::Vector3d> normals;
	std::vector<Sophus::SE3d> poses; // Absolute pose of each radar point
	Sophus::SE3d control_pose;
};

struct EulerAngles {
	double roll, pitch, yaw;
};

std::vector<RadarPoint> KStrong(cv::Mat img_in, size_t, double, double, double, double, double);

cv::Mat FormatDelete(cv::Mat);

cv::Mat RangeSelect(cv::Mat, double, double, double);

double FormatInfo(cv::Mat img_in);

std::vector<RadarPoint> MotionCompensation(cv::Mat, std::vector<RadarPoint>, std::vector<double>);

std::vector<RadarPoint> Downsample(std::vector<RadarPoint>, double, double, double);

Scan Represent(cv::Mat, std::vector<RadarPoint>, std::vector<RadarPoint>, int, double, double);

int FindNeighbor(Scan, Scan, int i, double grid_r, double yaw_max);

void SaveCartesianImage(Scan, double, double, std::string);

EulerAngles Quaternion2EulerAngles(double*);

#endif
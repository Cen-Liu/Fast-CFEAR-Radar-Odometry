/*-------------------------------------------------------------------------------------------------------------
Fast-CFEAR Algorithm Implementation
AUTHOR : LIU CEN - NUS ECE
VERSION: Consecutive Scan Registration + Constant Velocity Model
-------------------------------------------------------------------------------------------------------------*/
#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "test/ceres/local_parameterization_se3.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include "SE3_SPLINE.h"
#include "Functions.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Sophus;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


class Radar_Residual {
public:
	explicit Radar_Residual(double* T_wk_data, Eigen::Vector3d mu_ti, Eigen::Vector3d mu_kj, Eigen::Vector3d n_kj)
		: T_wk_data_(T_wk_data), mu_ti_(mu_ti), mu_kj_(mu_kj), n_kj_(n_kj) {}

	template <typename T>
	bool operator()(const T* const T_wt_data, T* residual) const {

		// Pose of Scan_k w.r.t. World
		const Eigen::Map<SE3d> T_wk((double*)&T_wk_data_[0]);

		// Pose of Scan_t w.r.t. World
		const Eigen::Map<SE3<T>> T_wt((T*)&T_wt_data[0]);

		// Pose of Scan_t w.r.t. Scan_k
		const Sophus::SE3<T> T_kt = T_wk.inverse() * T_wt;
		//const Eigen::Matrix<T, 3, 1> t = T_kt.matrix().block<3, 1>(0, 3);
		//const Eigen::Matrix<T, 3, 3> R = T_kt.matrix().block<3, 3>(0, 0);

		// Objective function
		//residual[0] = n_kj_.transpose() * (R * mu_ti_ + t - mu_kj_);

		residual[0] = n_kj_.transpose() * ((T_kt.matrix() * mu_ti_.homogeneous()).hnormalized() - mu_kj_);

		return true;
	}

	static ceres::CostFunction* Create(double* T_wk_data, Eigen::Vector3d mu_ti, Eigen::Vector3d mu_kj, Eigen::Vector3d n_kj) {
		return new ceres::AutoDiffCostFunction<Radar_Residual, 1, 7>(
			new Radar_Residual(T_wk_data, mu_ti, mu_kj, n_kj));
	}

private:
	const double* T_wk_data_;
	const Eigen::Vector3d mu_ti_, mu_kj_, n_kj_;
};


int main(int argc, char** argv) {
	//----------------------------------------- Read Radar Images -----------------------------------------
	std::vector<string> paths; // Path of each image

	// Oxford Radar Dataset
	const std::string img_path = "C:/Users/liucen/Documents/EE_Project/Dataset/2019-01-15-14-24-38-radar-oxford-10k/radar/*.png";
	const std::string save_path = "C:/Users/liucen/Documents/EE_Project/Dataset/2019-01-15-14-24-38-radar-oxford-10k/processed_output/";

	glob(img_path, paths); // Read image path

	// ------------------------------------ Parameters in CFEAR paper ------------------------------------
	// Set range to be considered. Note: All distances are measured in meter.
	const double range_down = 1.0; // Downsampling rate for the range of polar images: (0, 1]
	const double range_res = 0.0438 / range_down; // Range resolution [m]
	const double Dmin = 5.0;   // Min. sensor distance [m]
	const double Dmax = 100.0; // Max. sensor distance [m]

	// K-strongest filtering parameter
	const size_t K = 3;

	const double Zmin = 55; // Power threshold

	// Downsampling parameter
	const double grid_r = 3.5; // Side length of a grid [m]
	const double grid_f = 1.0; // Resample factor

	// Scan representation parameter
	const int inlier_num = 6; // Discard distributions whose inliers are less than this number
	const double condition_num = 1e5; // Discard ill-defined distributions whose condition number is bigger than this value

	// Keyframes initialization
	const int S = 1; // Number of keyframes

	std::deque<Scan> keyframes; // Keyframes
	Scan current_frame; // Current current_frame

	const double dist_keyframe = 1.5; // Admit current frame into keyframes if estimated translation exceeds this value [m]
	const double yaw_keyframe = 1.5 * (M_PI / 180); // Admit current frame into keyframes if estimated rotation exceeds this value [rads]

	//const double dist_keyframe = 0.0; // Two Consecutive Scans Registration
	//const double yaw_keyframe = 0.0;

	std::deque<Sophus::SE3d> T_wk_queue; // Poses of keyframes w.r.t. World
	Sophus::SE3d T_wt = Sophus::SE3d(); // Pose of current current_frame w.r.t. World
	Sophus::SE3d T_kt = Sophus::SE3d(); // Pose of current current_frame w.r.t. the latest keyframe

	std::vector<double> velocity(3); // Velocity of x, y and yaw

	double sum_iteration = 0.0;

	// The output file contains all frames
	std::ofstream ofile_allframes;
	ofile_allframes.open("C:/Users/liucen/Documents/EE_Project/Dataset/Est.txt");
	ofile_allframes << "Timestamp x y z qx qy qz qw" << endl;

	std::ofstream ofile_allframes_new;
	ofile_allframes_new.open("C:/Users/liucen/Documents/EE_Project/Dataset/Est_new.txt");

	// The output file only contains keyframes
	std::ofstream ofile_keyframes;
	ofile_keyframes.open("C:/Users/liucen/Documents/EE_Project/Dataset/Key.txt");
	ofile_keyframes << "Timestamp x y z qx qy qz qw" << endl;

	std::ofstream ofile_keyframes_new;
	ofile_keyframes_new.open("C:/Users/liucen/Documents/EE_Project/Dataset/Key_new.txt");

	google::InitGoogleLogging(argv[0]);

	for (int n = 0; n < paths.size(); n++) {
		// ------------------------------------- Image Preprocess -------------------------------------
		// Read grayscale image
		cv::Mat img_polar_oxford = cv::imread(paths[n], IMREAD_UNCHANGED);

		// Convert to double
		img_polar_oxford.convertTo(img_polar_oxford, CV_64FC1);

		// K-strongest filtering
		std::vector<RadarPoint> kstrong_pointset = KStrong(img_polar_oxford, K, Zmin, Dmin, Dmax, range_res, range_down);

		//// Motion compensation
		//std::vector<RadarPoint> kstrong_mc_pointset = MotionCompensation(img_polar_oxford, kstrong_pointset, velocity);

		// Downsample
		std::vector<RadarPoint> downsample_pointset = Downsample(kstrong_pointset, grid_r, grid_f, Dmax);

		// Scan representation
		Scan frame = Represent(img_polar_oxford, kstrong_pointset, downsample_pointset, inlier_num, condition_num, grid_r);

		//// Save frame representation as Cartesian images
		//SaveCartesianImage(frame, Dmax, range_res, save_path);

		// -------------------------------------- Initialization --------------------------------------
		if (n == 0) { // Keyframes initialization
			for (int i = 0; i < S; i++) {
				keyframes.push_back(frame);
				T_wk_queue.push_back(Sophus::SE3d());
			}
			ofile_allframes << static_cast<long long>(frame.timespan(0)) << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
			ofile_keyframes << static_cast<long long>(frame.timespan(0)) << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << endl;
			continue;
		}

		current_frame = frame;

		T_wt = T_wk_queue.back() * T_kt; // Initialization based on constant velocity model

		const double x_ini = T_wt.data()[4], y_ini = T_wt.data()[5], z_ini = T_wt.data()[6]; // Translation vector
		const double qx_ini = T_wt.data()[0], qy_ini = T_wt.data()[1], qz_ini = T_wt.data()[2], qw_ini = T_wt.data()[3]; // Quaternion

		// --------------------------------------- Optimization ---------------------------------------
		// Configure the loss function
		const double delta = 0.1;
		const double a = delta / sqrt(2); // Scaling factor of Huber loss function
		LossFunction* huber_loss_func = new ceres::HuberLoss(a);

		ceres::Problem odometry;

		// Add the residuals
		const double yaw_max = 30.0 * (M_PI / 180.0); // Tolerance of angle between surface normals [rads]
		for (int m = 0; m < S; m++) {
			for (int i = 0; i < current_frame.means.size(); i++) {
				Eigen::Vector3d mu_ti(current_frame.means[i]);

				int j = FindNeighbor(current_frame, keyframes[m], i, grid_r, yaw_max);
				if (j != -1) {
					Eigen::Vector3d mu_kj(keyframes[m].means[j]);
					Eigen::Vector3d n_kj(keyframes[m].normals[j]);

					ceres::CostFunction* radar_cost_func = Radar_Residual::Create(T_wk_queue[m].data(), mu_ti, mu_kj, n_kj);
					odometry.AddResidualBlock(radar_cost_func, huber_loss_func, T_wt.data());
				}
			}
		}

		// Build and solve the problem
		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.minimizer_type = ceres::LINE_SEARCH;
		options.line_search_direction_type = ceres::BFGS;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &odometry, &summary);

		sum_iteration += static_cast<double>(summary.iterations.size()) - 1.0;

		// Compute velocity
		T_kt = T_wk_queue.back().inverse() * T_wt;
		const double dist_kt = T_kt.translation().norm(); // Translation distance [m]
		const double yaw_kt = Quaternion2EulerAngles(T_kt.data()).yaw;

		double time_kt = (current_frame.timespan(1) - keyframes.back().timespan(1)) / 1000000.0; // [s]
		velocity[0] = T_kt.matrix()(0, 3) / time_kt;
		velocity[1] = T_kt.matrix()(1, 3) / time_kt;
		velocity[2] = yaw_kt / time_kt;

		// -------------------------------------- Display Results --------------------------------------
		std::cout.setf(ios::fixed);
		std::cout.precision(3);
		std::cout << n << ": ";

		// Write results of each current_frame into a file
		ofile_allframes << static_cast<long long>(frame.timespan(0)) << " " << T_wt.data()[4] << " " << T_wt.data()[5] << " " << T_wt.data()[6] << " " << T_wt.data()[0] << " " << T_wt.data()[1] << " " << T_wt.data()[2] << " " << T_wt.data()[3] << endl;

		ofile_allframes_new << T_wt.matrix()(0, 0) << " " << T_wt.matrix()(0, 1) << " " << T_wt.matrix()(0, 2) << " " << T_wt.matrix()(0, 3) << " " << T_wt.matrix()(1, 0) << " " << T_wt.matrix()(1, 1) << " " << T_wt.matrix()(1, 2) << " " << T_wt.matrix()(1, 3) << " " << T_wt.matrix()(2, 0) << " " << T_wt.matrix()(2, 1) << " " << T_wt.matrix()(2, 2) << " " << T_wt.matrix()(2, 3) << endl;

		// Update keyframes
		if (dist_kt > dist_keyframe || abs(yaw_kt) > yaw_keyframe) {

			keyframes.pop_front();
			keyframes.push_back(current_frame);
			T_wk_queue.pop_front();
			T_wk_queue.push_back(T_wt);

			// Write results of each keyframe into another file
			ofile_keyframes << static_cast<long long>(frame.timespan(0)) << " " << T_wt.data()[4] << " " << T_wt.data()[5] << " " << T_wt.data()[6] << " " << T_wt.data()[0] << " " << T_wt.data()[1] << " " << T_wt.data()[2] << " " << T_wt.data()[3] << endl;
			ofile_keyframes_new << n << " " << T_wt.matrix()(0, 0) << " " << T_wt.matrix()(0, 1) << " " << T_wt.matrix()(0, 2) << " " << T_wt.matrix()(0, 3) << " " << T_wt.matrix()(1, 0) << " " << T_wt.matrix()(1, 1) << " " << T_wt.matrix()(1, 2) << " " << T_wt.matrix()(1, 3) << " " << T_wt.matrix()(2, 0) << " " << T_wt.matrix()(2, 1) << " " << T_wt.matrix()(2, 2) << " " << T_wt.matrix()(2, 3) << endl;
			std::cout << "keyframe updated";
		}

		std::cout << endl << summary.BriefReport() << endl;
		std::cout << "Initial: x = " << x_ini << " m, y = " << y_ini << " m, z = " << z_ini << " m, qx = " << qx_ini << ", qy = " << qy_ini << ", qz = " << qz_ini << ", qw = " << qw_ini << endl;
		std::cout << "Final  : x = " << T_wt.data()[4] << " m, y = " << T_wt.data()[5] << " m, z = " << T_wt.data()[6] << " m, qx = " << T_wt.data()[0] << ", qy = " << T_wt.data()[1] << ", qz = " << T_wt.data()[2] << ", qw = " << T_wt.data()[3] << endl << endl;
	}

	std::cout << "The run time is: " << static_cast<double>(clock()) / CLOCKS_PER_SEC << "s" << endl;
	std::cout << "The average iteration is: " << sum_iteration / (static_cast<double>(paths.size()) - 1.0) << endl;
	ofile_allframes.close();
	ofile_keyframes.close();

	ofile_allframes_new.close();
	ofile_keyframes_new.close();

	return 0;
}
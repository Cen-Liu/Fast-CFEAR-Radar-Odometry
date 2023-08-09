#define _USE_MATH_DEFINES
#include <cmath>
#include "Functions.h"

using namespace std;

EulerAngles Quaternion2EulerAngles(double* data) {
    EulerAngles angles;

    double qx = data[0], qy = data[1], qz = data[2], qw = data[3];

    // Roll (x-axis rotation)
    double sinr_cosp = 2 * (qw * qx + qy * qz);
    double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // Use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // Yaw (z-axis rotation)
    double siny_cosp = 2 * (qw * qz + qx * qy);
    double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}
#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <stdio.h>
#include <Eigen/Core>
#include <unordered_map>
#include <pcl/common/io.h>
#include <cv_bridge/cv_bridge.h>

#define MIN_PS 15
#define LAYER_LIMIT 3
#define SMALL_EPS 1e-10
#define HASH_P 116101
#define MAX_N 10000000019

enum OT_STATE {UNKNOWN, MID_NODE, PLANE};

struct PnPData
{
  double x, y, z, u, v;
};

struct VPnPData
{
  double x, y, z, u, v;
  Eigen::Vector2d direction;
  Eigen::Vector2d direction_lidar;
  int number;
};

typedef Eigen::Matrix<double, 6, 1> Vector6d;
//存放每一个点的索引值和其对应的曲率
typedef struct PCURVATURE
{
  int index;
  float curvature;
} PCURVATURE;

typedef struct SinglePlane
{
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZ p_center;
  Eigen::Vector3d normal;
  int index;
} SinglePlane;

typedef struct Plane
{
  pcl::PointXYZINormal p_center;
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Matrix3d covariance;
  std::vector<Eigen::Vector3d> plane_points;
  float radius = 0;
  float min_eigen_value = 1;
  float d = 0;
  int points_size = 0;
  bool is_plane = false;
  bool is_init = false;
  int id;
  bool is_update = false;
} Plane;

Eigen::Matrix3d wedge(const Eigen::Vector3d& v)
{
	Eigen::Matrix3d V;
	V << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
	return V;
}

Eigen::Quaterniond exp(const Eigen::Vector3d& omega)
{
	double theta = omega.norm();
	double half_theta = 0.5 * theta;

	double imag_factor;
	double real_factor = cos(half_theta);
	if(theta < SMALL_EPS)
	{
		double theta_sq = theta * theta;
		double theta_po4 = theta_sq * theta_sq;
		imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
	}
	else
	{
		double sin_half_theta = sin(half_theta);
		imag_factor = sin_half_theta / theta;
	}

	return Eigen::Quaterniond(real_factor, imag_factor * omega.x(),
		                        imag_factor * omega.y(), imag_factor * omega.z());
}

void assign_qt(Eigen::Quaterniond& q, Eigen::Vector3d& t,
               Eigen::Quaterniond& q_, Eigen::Vector3d& t_)
{
  q.w() = q_.w(); q.x() = q_.x(); q.y() = q_.y(); q.z() = q_.z();
  t(0) = t_(0); t(1) = t_(1); t(2) = t_(2);
}

void assign_q(Eigen::Quaterniond& q, const Eigen::Quaterniond& q_)
{
  q.w() = q_.w(); q.x() = q_.x(); q.y() = q_.y(); q.z() = q_.z();
}

void assign_t(Eigen::Vector3d& t, const Eigen::Vector3d& t_)
{
  t(0) = t_(0); t(1) = t_(1); t(2) = t_(2);
}

double cos_angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  return (a.dot(b) / a.norm()) / b.norm();
}

using namespace std;

template <class T> void input(T matrix[4][5])
{
  cout << "please input matrix element's data" << endl;
  for(int i = 1; i < 4; i++)
    for(int j = 1; j < 5; j++)
      cin >> matrix[i][j];
  cout << "input ok";
}

template <class T> void calc(T matrix[4][5], Eigen::Vector3d &solution)
{
  T base_D = matrix[1][1] * matrix[2][2] * matrix[3][3] +
             matrix[2][1] * matrix[3][2] * matrix[1][3] +
             matrix[3][1] * matrix[1][2] * matrix[2][3]; //计算行列式
  base_D = base_D - (matrix[1][3] * matrix[2][2] * matrix[3][1] +
                     matrix[1][1] * matrix[2][3] * matrix[3][2] +
                     matrix[1][2] * matrix[2][1] * matrix[3][3]);

  if(base_D != 0)
  {
    T x_D = matrix[1][4] * matrix[2][2] * matrix[3][3] +
            matrix[2][4] * matrix[3][2] * matrix[1][3] +
            matrix[3][4] * matrix[1][2] * matrix[2][3];
    x_D = x_D - (matrix[1][3] * matrix[2][2] * matrix[3][4] +
                 matrix[1][4] * matrix[2][3] * matrix[3][2] +
                 matrix[1][2] * matrix[2][4] * matrix[3][3]);
    T y_D = matrix[1][1] * matrix[2][4] * matrix[3][3] +
            matrix[2][1] * matrix[3][4] * matrix[1][3] +
            matrix[3][1] * matrix[1][4] * matrix[2][3];
    y_D = y_D - (matrix[1][3] * matrix[2][4] * matrix[3][1] +
                 matrix[1][1] * matrix[2][3] * matrix[3][4] +
                 matrix[1][4] * matrix[2][1] * matrix[3][3]);
    T z_D = matrix[1][1] * matrix[2][2] * matrix[3][4] +
            matrix[2][1] * matrix[3][2] * matrix[1][4] +
            matrix[3][1] * matrix[1][2] * matrix[2][4];
    z_D = z_D - (matrix[1][4] * matrix[2][2] * matrix[3][1] +
                 matrix[1][1] * matrix[2][4] * matrix[3][2] +
                 matrix[1][2] * matrix[2][1] * matrix[3][4]);

    T x = x_D / base_D;
    T y = y_D / base_D;
    T z = z_D / base_D;
    // cout << "[ x:" << x << "; y:" << y << "; z:" << z << " ]" << endl;
    solution[0] = x;
    solution[1] = y;
    solution[2] = z;
  }
  else
  {
    cout << "【无解】";
    solution[0] = 0;
    solution[1] = 0;
    solution[2] = 0;
    //        return DBL_MIN;
  }
}

void rgb2grey(const cv::Mat &rgb_image, cv::Mat &grey_img) {
  for(int x = 0; x < rgb_image.cols; x++) {
    for(int y = 0; y < rgb_image.rows; y++) {
      grey_img.at<uchar>(y, x) = 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[0] +
                                 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[1] +
                                 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[2];
    }
  }
}

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if(v < vmin) {
    v = vmin;
  }

  if(v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if(v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if(v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if(v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if(v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}
typedef struct VoxelGrid {
  float size = 0.5;
  int index;
  Eigen::Vector3d origin;
  pcl::PointCloud<pcl::PointXYZI> cloud;
} VoxelGrid;

typedef struct Voxel {
  float size;
  Eigen::Vector3d voxel_origin;
  std::vector<uint8_t> voxel_color;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  Voxel(float _size) : size(_size) {
    voxel_origin << 0, 0, 0;
    cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
  };
} Voxel;

#endif
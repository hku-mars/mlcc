#include <fstream>
#include <iomanip>
#include <iostream>

#include "calib_single_camera.hpp"
#include "ceres/ceres.h"
#include "common.h"

// #define debug_mode

using namespace std;
using namespace Eigen;

Eigen::Matrix3d inner;
#ifdef FISHEYE
Eigen::Vector4d distor;
#else
Eigen::Matrix<double, 5, 1> distor;
#endif

class vpnp_calib
{
public:
  vpnp_calib(VPnPData p) {pd = p;}
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    #ifdef FISHEYE
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T a = p_c[0] / p_c[2];
    T b = p_c[1] / p_c[2];
    T r = sqrt(a * a + b * b);
    T theta = atan(r);
    T theta_d = theta *
      (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
        distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

    T dx = (theta_d / r) * a;
    T dy = (theta_d / r) * b;
    T ud = fx * dx + cx;
    T vd = fy * dy + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);
    #else
    Eigen::Matrix<T, 5, 1> distorT = distor.cast<T>();
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T& fx = innerT.coeffRef(0, 0);
    const T& cx = innerT.coeffRef(0, 2);
    const T& fy = innerT.coeffRef(1, 1);
    const T& cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4 + distorT[4] * r2 * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
            distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
            distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;

    if(T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0))
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    }
    else
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I = Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 0) = residuals[1];
      R = V * R;
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 0);
    }
    #endif
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p)
  {
    return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(new vpnp_calib(p)));
  }

private:
  VPnPData pd;
};

class vpnp_calib_rotation
{
public:
  vpnp_calib_rotation(VPnPData p, Vector3d trans_) {pd = p; trans = trans_;}
  template <typename T>
  bool operator()(const T *_q, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{T(trans(0)), T(trans(1)), T(trans(2))};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    #ifdef FISHEYE
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T a = p_c[0] / p_c[2];
    T b = p_c[1] / p_c[2];
    T r = sqrt(a * a + b * b);
    T theta = atan(r);
    T theta_d = theta * (T(1) + distorT[0] * pow(theta, T(2)) + distorT[1] * pow(theta, T(4)) +
      distorT[2] * pow(theta, T(6)) + distorT[3] * pow(theta, T(8)));

    T dx = (theta_d / r) * a;
    T dy = (theta_d / r) * b;
    T ud = fx * dx + cx;
    T vd = fy * dy + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);
    #else
    Eigen::Matrix<T, 5, 1> distorT = distor.cast<T>();
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T& fx = innerT.coeffRef(0, 0);
    const T& cx = innerT.coeffRef(0, 2);
    const T& fy = innerT.coeffRef(1, 1);
    const T& cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4 + distorT[4] * r2 * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
            distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
            distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;

    if(T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0))
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    }
    else
    {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I = Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;
      Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 0) = residuals[1];
      R = V * R;
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 0);
    }
    #endif
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p, Vector3d trans_)
  {
    return (new ceres::AutoDiffCostFunction<vpnp_calib_rotation, 2, 4>(new vpnp_calib_rotation(p, trans_)));
  }

private:
  VPnPData pd;
  Vector3d trans;
};

void roughCalib(Calibration& calibra, double search_resolution, int max_iter)
{
  float match_dis = 25;
  Eigen::Vector3d fix_adjust_euler(0, 0, 0);
  ROS_INFO_STREAM("roughCalib");
  for(int n = 0; n < 2; n++)
    for(int round = 0; round < 3; round++)
    {
      Eigen::Matrix3d rot = calibra.camera_.ext_R;
      Vector3d transation = calibra.camera_.ext_t;
      float min_cost = 1000;
      for(int iter = 0; iter < max_iter; iter++)
      {
        Eigen::Vector3d adjust_euler = fix_adjust_euler;
        adjust_euler[round] = fix_adjust_euler[round] + pow(-1, iter) * int(iter / 2) * search_resolution;
        Eigen::Matrix3d adjust_rotation_matrix;
        adjust_rotation_matrix =
          Eigen::AngleAxisd(adjust_euler[0], Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(adjust_euler[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(adjust_euler[2], Eigen::Vector3d::UnitX());
        Eigen::Matrix3d test_rot = rot * adjust_rotation_matrix;
        Eigen::Vector3d test_euler = test_rot.eulerAngles(2, 1, 0);
        Vector6d test_params;
        test_params << test_euler[0], test_euler[1], test_euler[2], transation[0], transation[1], transation[2];
        std::vector<VPnPData> pnp_list;
        calibra.buildVPnp(calibra.camera_, test_params, match_dis,
                          false, calibra.camera_.rgb_edge_cloud_,
                          calibra.lidar_edge_cloud_, pnp_list);

        int edge_size = calibra.lidar_edge_cloud_->size();
        int pnp_size = pnp_list.size();
        float cost = ((float)(edge_size - pnp_size) / (float)edge_size);
        #ifdef debug_mode
        std::cout << "n " << n << " round " << round << " a " << a << " iter "
                  << iter << " cost:" << cost << std::endl;
        #endif
        if(cost < min_cost)
        {
          ROS_INFO_STREAM("cost " << cost << " edge size "
                                  << calibra.lidar_edge_cloud_->size()
                                  << " pnp_list " << pnp_list.size());
          min_cost = cost;
          Eigen::Matrix3d rot;
          rot = Eigen::AngleAxisd(test_params[0], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(test_params[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(test_params[2], Eigen::Vector3d::UnitX());
          calibra.camera_.update_Rt(rot, transation);
          calibra.buildVPnp(calibra.camera_, test_params, match_dis,
                            true, calibra.camera_.rgb_edge_cloud_,
                            calibra.lidar_edge_cloud_, pnp_list);
          cv::Mat projection_img = calibra.getProjectionImg(test_params);
          cv::resize(projection_img, projection_img, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
          cv::imshow("rough calib", projection_img);
          cv::waitKey(10);
        }
      }
    }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "calib_single_camera");
  ros::NodeHandle nh;

  int dis_thr_low_bound;
  bool use_ada_voxel, use_rough_calib, only_calib_rotation;
  nh.getParam("distance_threshold_lower_bound", dis_thr_low_bound);
  nh.getParam("use_adaptive_voxel", use_ada_voxel);
  nh.getParam("use_rough_calib", use_rough_calib);
  nh.getParam("only_calibrate_rotation", only_calib_rotation);

  const string CamConfigPath = string(argv[1]);
  const string CalibSettingPath = string(argv[2]);
  const string ResultPath = string(argv[3]);

  /* load calibration configurations */
  Calibration calib(CamConfigPath, CalibSettingPath, use_ada_voxel);
  if(use_rough_calib) roughCalib(calib, DEG2RAD(0.1), 30);

  string result_file = ResultPath + "/extrinsic.txt";
  ofstream outfile;
  outfile.open(result_file, ofstream::trunc);
  outfile.close();

  /* calibration process */
  int iter = 0;
  ros::Time begin_t = ros::Time::now();
  for(int dis_threshold = 30; dis_threshold > dis_thr_low_bound; dis_threshold -= 1)
  {
    cout << "Iteration:" << iter++ << " Distance:" << dis_threshold << endl;
    for(int cnt = 0; cnt < 2; cnt++)
    {
      Eigen::Vector3d euler_angle = calib.camera_.ext_R.eulerAngles(2, 1, 0); // 2 is z, 0 is x
      Eigen::Vector3d transation = calib.camera_.ext_t;
      Vector6d calib_params;
      calib_params << euler_angle(0), euler_angle(1), euler_angle(2), transation(0), transation(1), transation(2);
      vector<VPnPData> vpnp_list;
      Eigen::Matrix3d R;
      Eigen::Vector3d T;
      R = calib.camera_.ext_R;
      T = calib.camera_.ext_t;
      
      inner << calib.camera_.fx_, calib.camera_.s_, calib.camera_.cx_,
               0, calib.camera_.fy_, calib.camera_.cy_,
               0, 0, 1;
      #ifdef FISHEYE
      distor << calib.camera_.k1_, calib.camera_.k2_, calib.camera_.k3_, calib.camera_.k4_;
      #else
      distor << calib.camera_.k1_, calib.camera_.k2_, calib.camera_.p1_, calib.camera_.p2_, calib.camera_.k3_;
      #endif

      calib.buildVPnp(calib.camera_, calib_params, dis_threshold, true,
                      calib.camera_.rgb_edge_cloud_, calib.lidar_edge_cloud_, vpnp_list);

      Eigen::Quaterniond q(R);
      double ext[7];
      ext[0] = q.x(); ext[1] = q.y(); ext[2] = q.z(); ext[3] = q.w();
      ext[4] = T[0]; ext[5] = T[1]; ext[6] = T[2];
      Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
      Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

      ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
      ceres::Problem problem;
      problem.AddParameterBlock(ext, 4, q_parameterization);
      problem.AddParameterBlock(ext + 4, 3);
      for(auto val: vpnp_list)
      {
        ceres::CostFunction* cost_function;
        if(only_calib_rotation)
        {
          cost_function = vpnp_calib_rotation::Create(val, T);
          problem.AddResidualBlock(cost_function, NULL, ext);
        }
        else
        {
          cost_function = vpnp_calib::Create(val);
          problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
        }
      }
      ceres::Solver::Options options;
      options.preconditioner_type = ceres::JACOBI;
      options.linear_solver_type = ceres::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = false;
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      #ifdef debug_mode
      cout << summary.BriefReport() << endl;
      #endif

      calib.camera_.update_Rt(m_q.toRotationMatrix(), m_t);
      vector<VPnPData>().swap(vpnp_list);
    }
    cout << endl;
  }
  ros::Time end_t = ros::Time::now();
  cout << "time taken " << (end_t-begin_t).toSec() << endl;

  /* output calibrated extrinsic results */
  Eigen::Vector3d euler_angle = calib.camera_.ext_R.eulerAngles(2, 1, 0);
  Eigen::Vector3d transation = calib.camera_.ext_t;
  Vector6d calib_params;
  calib_params << euler_angle(0), euler_angle(1), euler_angle(2), transation(0), transation(1), transation(2);
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  R = calib.camera_.ext_R;
  T = calib.camera_.ext_t;
  outfile.open(result_file, ofstream::app);
  for(int i = 0; i < 3; i++)
    outfile << R(i, 0) << "," << R(i, 1) << "," << R(i, 2) << "," << T[i] << "\n";
  outfile << 0 << "," << 0 << "," << 0 << "," << 1 << "\n";
  outfile.close();

  /* visualize the colorized point cloud */
  calib_params << euler_angle(0), euler_angle(1), euler_angle(2),
                  transation(0), transation(1), transation(2);
  calib.colorCloud(calib_params, 1, calib.camera_, calib.camera_.rgb_img_, calib.lidar_cloud_);

  while(ros::ok())
  {
    cout << "please reset rviz and push enter to publish again" << endl;
    getchar();
    Eigen::Vector3d euler_angle = calib.camera_.ext_R.eulerAngles(2, 1, 0);
    Eigen::Vector3d transation = calib.camera_.ext_t;
    Vector6d calib_params;
    calib_params << euler_angle(0), euler_angle(1), euler_angle(2), transation(0), transation(1), transation(2);
    calib.colorCloud(calib_params, 1, calib.camera_, calib.camera_.rgb_img_, calib.lidar_cloud_);
  }
  return 0;
}
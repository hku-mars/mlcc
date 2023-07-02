#include <fstream>
#include <iomanip>
#include <iostream>
#include "ceres/ceres.h"
#include "calib_camera.hpp"
#include "common.h"
// #define debug_mode

using namespace std;
using namespace Eigen;

Eigen::Matrix3d inner;
Eigen::Vector4d distor;
Eigen::Vector4d quaternion;
Eigen::Vector3d transation;

class pnp_calib
{
public:
  pnp_calib(PnPData p) {pd = p;}
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
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
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
            distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
            distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    residuals[0] = ud - T(pd.u);
    residuals[1] = vd - T(pd.v);
    return true;
  }
  static ceres::CostFunction *Create(PnPData p)
  {
    return (new ceres::AutoDiffCostFunction<pnp_calib, 2, 4, 3>(new pnp_calib(p)));
  }

private:
  PnPData pd;
};

class vpnp_calib
{
public:
  vpnp_calib(VPnPData p) {pd = p;}
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
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
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
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
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p)
  {
    return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(new vpnp_calib(p)));
  }

private:
  VPnPData pd;
};

void roughCalib(Calibration &calibra, double search_resolution, int max_iter)
{
  float match_dis = 20;
  Eigen::Vector3d fix_adjust_euler(0, 0, 0);
  ROS_INFO_STREAM("roughCalib");
  for(int n = 0; n < 2; n++)
  {
    for(int round = 0; round < 3; round++)
    {
      for(size_t a = 0; a < calibra.cams.size(); a++)
      {
        Eigen::Matrix3d rot = calibra.cams[a].ext_R;
        Vector3d transation = calibra.cams[a].ext_t;
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
          calibra.buildVPnp(calibra.cams[a], test_params, match_dis,
                            false, calibra.cams[a].rgb_edge_clouds,
                            calibra.lidar_edge_clouds, pnp_list);

          int edge_size = calibra.lidar_edge_clouds->size();
          int pnp_size = pnp_list.size();
          float cost = ((float)(edge_size - pnp_size) / (float)edge_size);
          #ifdef debug_mode
          std::cout << "n " << n << " round " << round << " a " << a << " iter "
                    << iter << " cost:" << cost << std::endl;
          #endif
          if(cost < min_cost)
          {
            ROS_INFO_STREAM("cost " << cost << " edge size "
                                    << calibra.lidar_edge_clouds->size()
                                    << " pnp_list " << pnp_list.size());
            min_cost = cost;
            Eigen::Matrix3d rot;
            rot = Eigen::AngleAxisd(test_params[0], Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(test_params[1], Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(test_params[2], Eigen::Vector3d::UnitX());
            calibra.cams[a].update_Rt(rot, transation);
            calibra.buildVPnp(calibra.cams[a], test_params, match_dis,
                              true, calibra.cams[a].rgb_edge_clouds,
                              calibra.lidar_edge_clouds, pnp_list);
            // cv::Mat projection_img = calibra.getProjectionImg(test_params, a, 0);
            // std::string img_name = std::to_string(a) + "_projection";
            // cv::imshow(img_name, projection_img);
            // cv::waitKey(10);
          }
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "calib_camera");
  ros::NodeHandle nh;

  int dis_thr_low_bound;
  bool use_adaptive_voxel, use_rough_calib;

  nh.getParam("distance_threshold_lower_bound", dis_thr_low_bound);
  nh.getParam("use_adaptive_voxel", use_adaptive_voxel);
  nh.getParam("use_rough_calib", use_rough_calib);

  const string LeftCamCfgPath = string(argv[1]);
  const string RightCamCfgPath = string(argv[2]);
  const string CalibSettingPath = string(argv[3]);
  const string ResultPath = string(argv[4]);

  /* load calibration configurations */
  vector<string> CamCfgPaths;
  CamCfgPaths.emplace_back(LeftCamCfgPath);
  CamCfgPaths.emplace_back(RightCamCfgPath);
  Calibration calib(CamCfgPaths, CalibSettingPath, use_adaptive_voxel);
  if(use_rough_calib) roughCalib(calib, DEG2RAD(0.1), 30);

  /* calibration process */
  int iter = 0;
  ros::Time begin_t = ros::Time::now();
  for(int dis_threshold = 20; dis_threshold > dis_thr_low_bound; dis_threshold -= 1)
  {
    cout << "Iteration:" << iter++ << " Distance:" << dis_threshold << endl;
    for(int cnt = 0; cnt < 2; cnt++)
      for(size_t a = 0; a < calib.cams.size(); a++)
      {
        Eigen::Vector3d euler_angle = calib.cams[a].ext_R.eulerAngles(2, 1, 0); // 2 is z, 0 is x
        Eigen::Vector3d transation = calib.cams[a].ext_t;
        Vector6d calib_params;
        calib_params << euler_angle(0), euler_angle(1), euler_angle(2), transation(0), transation(1), transation(2);
        vector<VPnPData> vpnp_list;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        R = calib.cams[a].ext_R;
        T = calib.cams[a].ext_t;
        inner << calib.cams[a].fx_, calib.cams[a].s_, calib.cams[a].cx_, 0.0,
                 calib.cams[a].fy_, calib.cams[a].cy_, 0.0, 0.0, 1.0;
        distor << calib.cams[a].k1_, calib.cams[a].k2_, calib.cams[a].p1_, calib.cams[a].p2_;

        calib.buildVPnp(calib.cams[a], calib_params, dis_threshold, true,
                        calib.cams[a].rgb_edge_clouds, calib.lidar_edge_clouds, vpnp_list);

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
          cost_function = vpnp_calib::Create(val);
          problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
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

        calib.cams[a].update_Rt(m_q.toRotationMatrix(), m_t);
        vector<VPnPData>().swap(vpnp_list);
      }
  }
  ros::Time end_t = ros::Time::now();
  cout << "time taken " << (end_t-begin_t).toSec() << endl;

  /* output calibrated extrinsic results */
  string result_file = ResultPath + "/extrinsic.txt";
  ofstream outfile;
  outfile.open(result_file, ofstream::trunc);
  outfile.close();
  for(size_t a = 0; a < calib.cams.size(); a++)
  {
    Eigen::Vector3d euler_angle = calib.cams[a].ext_R.eulerAngles(2, 1, 0);
    Eigen::Vector3d transation = calib.cams[a].ext_t;
    Vector6d calib_params;
    calib_params << euler_angle(0), euler_angle(1), euler_angle(2),
                    transation(0), transation(1), transation(2);
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    R = calib.cams[a].ext_R;
    T = calib.cams[a].ext_t;
    outfile.open(result_file, ofstream::app);
    for(int i = 0; i < 3; i++)
      outfile << R(i, 0) << "," << R(i, 1) << "," << R(i, 2) << "," << T[i] << "\n";
    outfile << 0 << "," << 0 << "," << 0 << "," << 1 << "\n";
    outfile.close();
  }

  /* ground truth calculated from the chessboard method */
  Eigen::Matrix3d Rgt;
  Rgt << 0.999824645293243, -0.0183851254185105, 0.00355890820136192,
         0.0183851559551168, 0.999830978444791, 2.41378898009210e-05,
         -0.00355875044729421, 4.12974252036677e-05, 0.999993666774833;
  Matrix3d R_;
  R_ = Rgt.inverse() * (calib.cams[1].ext_R * calib.cams[0].ext_R.inverse());
  // Vector3d t_e = R_.eulerAngles(0, 1, 2);
  // cout << "Euler " << t_e.transpose() * 57.3 << endl;
  Eigen::Quaterniond q2(Rgt.transpose());
  Eigen::Quaterniond qme(calib.cams[1].ext_R * calib.cams[0].ext_R.inverse());
  // cout << qme.toRotationMatrix() << endl;
  cout << "angular error " << qme.angularDistance(q2) * 57.3 << " degree" << endl;
  cout << "baseline error "
       << fabs(calib.cams[0].ext_t(0)-(qme*calib.cams[1].ext_t)(0)-0.1072) << " m" << endl;

  /* visualize the colorized point cloud */
  Eigen::Vector3d euler_angle = calib.cams[0].ext_R.eulerAngles(2, 1, 0);
  Eigen::Vector3d transation = calib.cams[0].ext_t;
  Vector6d calib_params;
  calib_params << euler_angle(0), euler_angle(1), euler_angle(2),
                  transation(0), transation(1), transation(2);
  calib.colorCloud(calib_params, 1, calib.cams[0], calib.cams[0].rgb_imgs, calib.base_clouds);

  while(ros::ok())
  {
    cout << "please reset rviz and push enter to publish again" << endl;
    getchar();
    Eigen::Vector3d euler_angle = calib.cams[0].ext_R.eulerAngles(2, 1, 0);
    Eigen::Vector3d transation = calib.cams[0].ext_t;
    Vector6d calib_params;
    calib_params << euler_angle(0), euler_angle(1), euler_angle(2),
                    transation(0), transation(1), transation(2);
    calib.colorCloud(calib_params, 1, calib.cams[0], calib.cams[0].rgb_imgs, calib.base_clouds);
  }
  return 0;
}
#ifndef HIERARCHICAL_BA
#define HIERARCHICAL_BA

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tools.hpp"
#include "common.h"

#define WIN_SIZE 10
#define GAP 5
#define AVG_THR
#define FULL_HESS
#define ENABLE_RVIZ
// #define ENABLE_FILTER
// #define HESSIAN_USE_CURVE

const double one_three = (1.0 / 3.0);

int layer_limit = 3;
int MIN_PT = 30;
int thd_num = 16;

int BINGO_CNT = 0;
class OCTO_TREE_NODE
{
public:
  OT_STATE octo_state;
  PLV(3) vec_orig;
  VOX_FACTOR sig_orig;

  OCTO_TREE_NODE* leaves[8];
  Plane* plane_ptr;
  float voxel_center[3];
  float quater_length;
  float eigen_thr;
  int layer;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;
  
  #ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_center = nh.advertise<sensor_msgs::PointCloud2>("/child_center", 1000);
  ros::Publisher pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("/child_voxel", 1000);
  ros::Publisher pub_cloud0 = nh.advertise<sensor_msgs::PointCloud2>("/child_cloud0", 1000);
  ros::Publisher pub_cloud1 = nh.advertise<sensor_msgs::PointCloud2>("/child_cloud1", 1000);
  ros::Publisher pub_cloud2 = nh.advertise<sensor_msgs::PointCloud2>("/child_cloud2", 1000);
  ros::Publisher pub_cloud3 = nh.advertise<sensor_msgs::PointCloud2>("/child_cloud3", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/child_direct", 1000);
  #endif

  OCTO_TREE_NODE(float _eigen_thr = 1.0/10): eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN;
    layer = 0;
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen(int layer)
  {
    VOX_FACTOR covMat = sig_orig;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest

    if(eigen_ratio > eigen_thr) return 0;
    if(saes.eigenvalues()[0] / saes.eigenvalues()[1] > 0.1) return 0; // 排除线状点云
    
    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    vector<VOX_FACTOR> covMats(8);

    for(Eigen::Vector3d ap: vec_orig)
    {
      int xyz[3] = {0, 0, 0};
      for(int k = 0; k < 3; k++)
        if(ap(k) > center_turb[k])
          xyz[k] = 1;

      Eigen::Vector3d pvec(ap(0), ap(1), ap(2));
      
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      covMats[leafnum].push(pvec);
    }
    
    int num_all = 0, num_qua = 0;
    for(int i = 0; i < 8; i++)
      if(covMats[i].N > MIN_PT)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        Eigen::Vector3d child_direct = saes.eigenvectors().col(0);

        if(fabs(child_direct.dot(direct)) > 0.98)
          num_qua++;
        num_all++;
      }
    
    if(num_qua != num_all) return 0;
    return 1;
  }

  void cut_func()
  {
    PLV(3)& pvec_orig = vec_orig;
    uint a_size = pvec_orig.size();

    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_orig[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }
      /*原始点云信息*/
      leaves[leafnum]->vec_orig.push_back(pvec_orig[j]);
      leaves[leafnum]->sig_orig.push(pvec_orig[j]);
    }
    PLV(3)().swap(pvec_orig);
  }

  void recut()
  {
    if(octo_state == UNKNOWN)
    {
      int point_size = sig_orig.N;
      
      if(point_size < MIN_PT)
      {
        octo_state = MID_NODE;
        PLV(3)().swap(vec_orig);
        return;
      }

      if(judge_eigen(layer))
      {
        octo_state = PLANE;

        plane_ptr = new Plane;
        VOX_FACTOR covMat = sig_orig;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
        value_vector = saes.eigenvalues();
        center = covMat.v / covMat.N;
        direct = saes.eigenvectors().col(0);

        plane_ptr->covariance = covMat.cov();
        plane_ptr->center = center;
        plane_ptr->normal = direct;
        plane_ptr->radius = sqrt(value_vector[2]);
        plane_ptr->min_eigen_value = value_vector[0];
        plane_ptr->d = -direct.dot(center);
        plane_ptr->p_center.x = center(0);
        plane_ptr->p_center.y = center(1);
        plane_ptr->p_center.z = center(2);
        plane_ptr->p_center.normal_x = direct(0);
        plane_ptr->p_center.normal_y = direct(1);
        plane_ptr->p_center.normal_z = direct(2);
        plane_ptr->points_size = point_size;
        plane_ptr->is_plane = true;
        for(auto pt: vec_orig)
          plane_ptr->plane_points.push_back(pt);

        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          octo_state = MID_NODE;
          PLV(3)().swap(vec_orig);
          return;
        }
        cut_func();
      }
    }
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut();
  }

  void get_plane_list(std::vector<Plane*>& plane_list)
  {
    if(octo_state == PLANE)
      plane_list.push_back(plane_ptr);
    else
      if(layer <= layer_limit)
        for(int i = 0; i < 8; i++)
          if(leaves[i] != nullptr)
            leaves[i]->get_plane_list(plane_list);
  }

  void tras_display(pcl::PointCloud<pcl::PointXYZINormal>& color_cloud,
                    visualization_msgs::MarkerArray& marker_array,
                    int layer = 0)
  {
    float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

    if(octo_state == PLANE)
    {
      for(size_t j = 0; j < vec_orig.size(); j++)
      {
        Eigen::Vector3d& pvec = vec_orig[j];
        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z();
        ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
        ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
        ap.normal_z = sqrt(value_vector[0]);
        color_cloud.push_back(ap);
      }

      #ifdef ENABLE_RVIZ
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "basic_shapes";
      marker.id = BINGO_CNT; BINGO_CNT++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.color.a = 1;
      marker.color.r = (layer==0)?1:0; // 第一层voxel为红色
      marker.color.g = (layer==1)?1:0; // 第二层voxel为绿色
      marker.color.b = (layer==2)?1:0; // 第三层voxel为蓝色
      if(layer > 2)
      { // 第四层voxel为白色
        marker.color.r = 1;
        marker.color.g = 1;
        marker.color.b = 1;
      }
      marker.scale.x = 0.01;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.lifetime = ros::Duration();
      geometry_msgs::Point apoint;
      apoint.x = center(0); apoint.y = center(1); apoint.z = center(2);
      marker.points.push_back(apoint);
      apoint.x += 0.2*direct(0); apoint.y += 0.2*direct(1); apoint.z += 0.2*direct(2);
      marker.points.push_back(apoint);
      marker_array.markers.push_back(marker);
      #endif
    }
    else
    {
      if(layer == layer_limit) return;
      layer++;
      for(int i = 0; i < 8; i++)
        if(leaves[i] != nullptr)
          leaves[i]->tras_display(color_cloud, marker_array, layer);
    }
  }
};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(float _eigen_thr): OCTO_TREE_NODE(_eigen_thr){}
  PLV(3) all_points;
};

#endif
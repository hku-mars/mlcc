#include <iostream>
#include <fstream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>

#include "BA/mypcl.hpp"
#include "pose_refine.hpp"

using namespace std;
using namespace Eigen;
double voxel_size, eigen_thr;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*>& feature_map,
               pcl::PointCloud<PointType>::Ptr feature_pts,
               Eigen::Quaterniond q, Eigen::Vector3d t, int f_head, int window_size, double eigen_threshold)
{
	uint pt_size = feature_pts->size();
	for(uint i = 0; i < pt_size; i++)
	{
		PointType& pt = feature_pts->points[i];
		Eigen::Vector3d pt_origin(pt.x, pt.y, pt.z);
		Eigen::Vector3d pt_trans = q * pt_origin + t;

		float loc_xyz[3];
		for(int j = 0; j < 3; j++)
		{
			loc_xyz[j] = pt_trans[j] / voxel_size;
			if(loc_xyz[j] < 0)
				loc_xyz[j] -= 1.0;
		}

		VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
		auto iter = feature_map.find(position);
		if(iter != feature_map.end())
		{
			iter->second->origin_pc[f_head]->push_back(pt_origin);
			iter->second->transform_pc[f_head]->push_back(pt_trans);
		}
		else
		{
			OCTO_TREE* ot = new OCTO_TREE(window_size, eigen_threshold);
			ot->origin_pc[f_head]->push_back(pt_origin);
			ot->transform_pc[f_head]->push_back(pt_trans);

			ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
			ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
			ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
			ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
			feature_map[position] = ot;
		}
	}
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_refine");
  ros::NodeHandle nh("~");

  ros::Publisher pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf", 100);
  ros::Publisher pub_surf_debug = nh.advertise<sensor_msgs::PointCloud2>("/debug_surf", 100);

  string data_path;
  int max_iter, base_lidar;
  double downsmp_base;
  bool load_original = true;

  nh.getParam("data_path", data_path);
  nh.getParam("max_iter", max_iter);
  nh.getParam("base_lidar", base_lidar);
  nh.getParam("voxel_size", voxel_size);
  nh.getParam("eigen_threshold", eigen_thr);
  nh.getParam("downsample_base", downsmp_base);
  nh.getParam("load_original", load_original);

  sensor_msgs::PointCloud2 debugMsg, colorCloudMsg;
  vector<mypcl::pose> pose_vec;
  if(load_original)
    pose_vec = mypcl::read_pose(data_path + "original_pose/" + to_string(base_lidar) + ".json");
  else
    pose_vec = mypcl::read_pose(data_path + "pose.json");
  size_t pose_size = pose_vec.size();
  ros::Time t_begin, t_end, cur_t;
  double avg_time = 0.0;

  pcl::PointCloud<PointType>::Ptr pc_debug(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pc_surf(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pc_full(new pcl::PointCloud<PointType>);

  vector<pcl::PointCloud<PointType>::Ptr> base_pc, ref_pc;
  base_pc.resize(pose_size);

  for(size_t i = 0; i < pose_size; i++)
  {
    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile(data_path+to_string(base_lidar)+"/"+to_string(i)+".pcd", *pc);
    base_pc[i] = pc;
  }
  
  int loop = 0;
  for(; loop < max_iter; loop++)
  {
    cout << "---------------------" << endl;
    cout << "iteration " << loop << endl;
    t_begin = ros::Time::now();
    int window_size = pose_size;
    unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map;
    LM_OPTIMIZER lm_opt(window_size);
    cur_t = ros::Time::now();

    for(size_t i = 0; i < pose_size; i++)
    {
      if(downsmp_base > 0) downsample_voxel(*base_pc[i], downsmp_base);
      cut_voxel(surf_map, base_pc[i], pose_vec[i].q, pose_vec[i].t, i, window_size, eigen_thr);
    }

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();

    for(int i = 0; i < window_size; i++)
      assign_qt(lm_opt.poses[i], lm_opt.ts[i], pose_vec[i].q, pose_vec[i].t);
    
    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->feed_pt(lm_opt);

    lm_opt.optimize();

    for(int i = 0; i < window_size; i++)
      assign_qt(pose_vec[i].q, pose_vec[i].t, lm_opt.poses[i], lm_opt.ts[i]);

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    
    t_end = ros::Time::now();
    cout << "time cost " << (t_end-t_begin).toSec() << endl;
    avg_time += (t_end-t_begin).toSec();

    Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
    Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
    pc_full->clear();
    for(size_t i = 0; i < pose_size; i++)
    {
      mypcl::transform_pointcloud(*base_pc[i], *pc_debug, q0.inverse()*(pose_vec[i].t-t0), q0.inverse()*pose_vec[i].q);
      pc_full = mypcl::append_cloud(pc_full, *pc_debug);
    }
    pcl::toROSMsg(*pc_full, debugMsg);
    debugMsg.header.frame_id = "camera_init";
    debugMsg.header.stamp = cur_t;
    pub_surf_debug.publish(debugMsg);
  }
  cout << "---------------------" << endl;
  cout << "complete" << endl;
  cout << "averaged iteration time " << avg_time / (loop+1) << endl;
  mypcl::write_pose(pose_vec, data_path);
  
  Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(),
                        pose_vec[0].q.y(), pose_vec[0].q.z());
  Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
  for(size_t i = 0; i < pose_size; i++)
  {
    pcl::io::loadPCDFile(data_path+to_string(base_lidar)+"/"+to_string(i)+".pcd", *pc_surf);
    mypcl::transform_pointcloud(*pc_surf, *pc_surf, q0.inverse()*(pose_vec[i].t-t0), q0.inverse()*pose_vec[i].q);

    pcl::toROSMsg(*pc_surf, colorCloudMsg);
    colorCloudMsg.header.frame_id = "camera_init";
    colorCloudMsg.header.stamp = cur_t;
    pub_surf.publish(colorCloudMsg);
  }

  ros::Rate loop_rate(1);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
}
#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <unordered_map>

#include "global.hpp"
#include "BA/mypcl.hpp"
#include "BA/tools.hpp"

using namespace std;
using namespace Eigen;
double voxel_size, eigen_thr;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE*>& feature_map,
               pcl::PointCloud<PointType>::Ptr feature_pts,
               Eigen::Quaterniond q, Eigen::Vector3d t, int f_head, int baselidar_sz,
               int exlidar_sz, double eigen_threshold, int exlidar_n = 0, bool is_base_lidar = true)
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
      if(is_base_lidar)
      {
        iter->second->baseOriginPc[f_head]->emplace_back(pt_origin);
        iter->second->baseTransPc[f_head]->emplace_back(pt_trans);
      }
      else
      {
        iter->second->refOriginPc[exlidar_n][f_head]->emplace_back(pt_origin);
        iter->second->refTransPc[exlidar_n][f_head]->emplace_back(pt_trans);
      }
		}
		else
		{
      OCTO_TREE *ot = new OCTO_TREE(baselidar_sz, exlidar_sz, eigen_threshold);
      if(is_base_lidar)
      {
        ot->baseOriginPc[f_head]->emplace_back(pt_origin);
        ot->baseTransPc[f_head]->emplace_back(pt_trans);
      }
      else
      {
        ot->refOriginPc[exlidar_n][f_head]->emplace_back(pt_origin);
        ot->refTransPc[exlidar_n][f_head]->emplace_back(pt_trans);
      }

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
  ros::init(argc, argv, "global_refine");
  ros::NodeHandle nh("~");

  ros::Publisher pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf", 100);
  ros::Publisher pub_surf1 = nh.advertise<sensor_msgs::PointCloud2>("/map_surf1", 100);
  ros::Publisher pub_surf2 = nh.advertise<sensor_msgs::PointCloud2>("/map_surf2", 100);
  ros::Publisher pub_surf3 = nh.advertise<sensor_msgs::PointCloud2>("/map_surf3", 100);

  string data_path, log_path;
  int max_iter, base_lidar, ref_lidar1, ref_lidar2, ref_lidar3 = -1;
  vector<int> ref_lidar;
  double downsmp_base, downsmp_ref;

  nh.getParam("data_path", data_path);
  nh.getParam("log_path", log_path);
  nh.getParam("max_iter", max_iter);
  nh.getParam("base_lidar", base_lidar);
  nh.getParam("ref_lidar1", ref_lidar1); ref_lidar.emplace_back(ref_lidar1);
  nh.getParam("ref_lidar2", ref_lidar2); ref_lidar.emplace_back(ref_lidar2);
  if(ref_lidar3 != -1) {nh.getParam("ref_lidar3", ref_lidar3); ref_lidar.emplace_back(ref_lidar3);}
  nh.getParam("voxel_size", voxel_size);
  nh.getParam("eigen_threshold", eigen_thr);
  nh.getParam("downsample_base", downsmp_base);
  nh.getParam("downsample_ref", downsmp_ref);

  sensor_msgs::PointCloud2 debugMsg, colorCloudMsg;
  vector<mypcl::pose> pose_vec = mypcl::read_pose(data_path + "pose.json");
  vector<mypcl::pose> ref_vec = mypcl::read_pose(data_path + "ref.json");
  size_t ref_size = ref_vec.size();
  size_t pose_size = pose_vec.size();
  
  ros::Time t_begin, t_end, cur_t;
  double avg_time = 0.0;

  pcl::PointCloud<PointType>::Ptr pc_debug(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pc_surf(new pcl::PointCloud<PointType>);

  vector<pcl::PointCloud<PointType>::Ptr> base_pc, ref_pc;
  base_pc.resize(pose_size);
  ref_pc.resize(pose_size * ref_size);
  for(size_t i = 0; i < pose_size; i++)
  {
    pcl::PointCloud<PointType>::Ptr pc_base(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile(data_path+to_string(base_lidar)+"/"+to_string(i)+".pcd", *pc_base);
    base_pc[i] = pc_base;
  }
  for(size_t i = 0; i < ref_size; i++)
    for(size_t j = 0; j < pose_size; j++)
    {
      pcl::PointCloud<PointType>::Ptr pc_ref(new pcl::PointCloud<PointType>);
      pcl::io::loadPCDFile(data_path+to_string(ref_lidar[i])+"/"+to_string(j)+".pcd", *pc_ref);
      ref_pc[i*pose_size+j] = pc_ref;
    }

  int loop = 0;
  for(; loop < max_iter; loop++)
  {
    cout << "---------------------" << endl;
    cout << "iteration " << loop << endl;
    t_begin = ros::Time::now();
    unordered_map<VOXEL_LOC, OCTO_TREE*> surf_map;
    LM_OPTIMIZER lm_opt(pose_size, ref_size);
    cur_t = ros::Time::now();

    for(size_t i = 0; i < pose_size; i++)
    {
      downsample_voxel(*base_pc[i], downsmp_base);
      for(int j = 0; j < ref_size; j++)
        downsample_voxel(*ref_pc[j*pose_size+i], downsmp_ref);
      
      cut_voxel(surf_map, base_pc[i], pose_vec[i].q, pose_vec[i].t, i, pose_size, ref_size, eigen_thr);

      for(size_t j = 0; j < ref_size; j++)
        cut_voxel(surf_map, ref_pc[j*pose_size+i], pose_vec[i].q * ref_vec[j].q,
                  pose_vec[i].q * ref_vec[j].t + pose_vec[i].t, i, pose_size, ref_size, eigen_thr, j, false);
    }

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();

    for(size_t i = 0; i < pose_size; i++)
      assign_qt(lm_opt.poses[i], lm_opt.ts[i], pose_vec[i].q, pose_vec[i].t);

    for(size_t i = 0; i < ref_size; i++)
      assign_qt(lm_opt.refQs[i], lm_opt.refTs[i], ref_vec[i].q, ref_vec[i].t);

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->feed_pt(lm_opt);
    
    lm_opt.optimize();

    for(size_t i = 0; i < pose_size; i++)
      assign_qt(pose_vec[i].q, pose_vec[i].t, lm_opt.poses[i], lm_opt.ts[i]);

    for(size_t i = 0; i < ref_size; i++)
      assign_qt(ref_vec[i].q, ref_vec[i].t, lm_opt.refQs[i], lm_opt.refTs[i]);

    for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;
    t_end = ros::Time::now();
    cout << "time cost " << (t_end-t_begin).toSec() << endl;
    avg_time += (t_end-t_begin).toSec();
  }

  cout << "---------------------" << endl;
  cout << "complete" << endl;
  cout << "averaged iteration time " << avg_time / (loop+1) << endl;
  mypcl::write_pose(pose_vec, ref_vec, data_path);

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
    {
      int j = 0;
      pcl::io::loadPCDFile(data_path+to_string(ref_lidar[j])+"/"+to_string(i)+".pcd", *pc_surf);
      mypcl::transform_pointcloud(*pc_surf, *pc_surf,
        q0.inverse()*(pose_vec[i].t-t0)+q0.inverse()*pose_vec[i].q*ref_vec[j].t,
        q0.inverse()*pose_vec[i].q*ref_vec[j].q);
      pcl::toROSMsg(*pc_surf, colorCloudMsg);
      colorCloudMsg.header.frame_id = "camera_init";
      colorCloudMsg.header.stamp = cur_t;
      pub_surf1.publish(colorCloudMsg);
    }
    {
      int j = 1;
      pcl::io::loadPCDFile(data_path+to_string(ref_lidar[j])+"/"+to_string(i)+".pcd", *pc_surf);
      mypcl::transform_pointcloud(*pc_surf, *pc_surf,
        q0.inverse()*(pose_vec[i].t-t0)+q0.inverse()*pose_vec[i].q*ref_vec[j].t,
        q0.inverse()*pose_vec[i].q*ref_vec[j].q);
      pcl::toROSMsg(*pc_surf, colorCloudMsg);
      colorCloudMsg.header.frame_id = "camera_init";
      colorCloudMsg.header.stamp = cur_t;
      pub_surf2.publish(colorCloudMsg);
    }
    if(ref_lidar3 != -1)
    {
      int j = 2;
      pcl::io::loadPCDFile(data_path+to_string(ref_lidar[j])+"/"+to_string(i)+".pcd", *pc_surf);
      mypcl::transform_pointcloud(*pc_surf, *pc_surf,
        q0.inverse()*(pose_vec[i].t-t0)+q0.inverse()*pose_vec[i].q*ref_vec[j].t,
        q0.inverse()*pose_vec[i].q*ref_vec[j].q);
      pcl::toROSMsg(*pc_surf, colorCloudMsg);
      colorCloudMsg.header.frame_id = "camera_init";
      colorCloudMsg.header.stamp = cur_t;
      pub_surf3.publish(colorCloudMsg);
    }
  }

  ros::Rate loop_rate(10);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
}
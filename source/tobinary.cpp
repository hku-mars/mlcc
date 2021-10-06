#include <iostream>
#include <fstream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include "ros/ros.h"

#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "json2binary");
    ros::NodeHandle nh("~");

    ros::Publisher pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf", 100);
    ros::Publisher pub_surf_debug = nh.advertise<sensor_msgs::PointCloud2>("/debug_surf", 100);

    string data_path, filename;
    int base_lidar;

    nh.getParam("data_path", data_path);
    nh.getParam("base_lidar", base_lidar);

    vector<mypcl::pose> pose_vec = mypcl::read_pose(data_path + "pose.json");
    size_t pose_size = pose_vec.size();

    vector<pcl::PointCloud<PointType>::Ptr> base_pc;
    base_pc.resize(pose_size);
    for (size_t i = 0; i < pose_size; i++)
    {
        cout<<"reading pointcloud "<<i<<endl;
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        *pc = mypcl::read_pointcloud(data_path + to_string(base_lidar) + 
            "/" + to_string(i) + ".json");
        base_pc[i] = pc;
    }

    for (size_t i = 0; i < pose_size; i++)
    {
        filename = data_path + to_string(base_lidar) + "/patch" + to_string(i) + ".dat";
        ofstream outFile(filename, ios::out | ios::binary);
        for(size_t j = 0; j < base_pc[i]->size(); j ++ )
        {
            mypcl::mypoint p(base_pc[i]->points[j].x, base_pc[i]->points[j].y, base_pc[i]->points[j].z);
            outFile.write((char*)&p, sizeof(p));
        }
        outFile.close();
    }
    
    // ifstream inFile(filename, ios::in | ios::binary);
    // mypcl::mypoint p;
    // while(inFile.read((char *)&p, sizeof(p)))
    // {
    //     int readedBytes = inFile.gcount();
    // }
    // inFile.close();
}
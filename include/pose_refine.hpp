#ifndef POSE_REFINE_HPP
#define POSE_REFINE_HPP

#include <mutex>
#include <cmath>
#include <thread>
#include <fstream>
#include <ros/ros.h>
#include <unordered_map>
#include <Eigen/StdVector>
#include <opencv2/imgproc.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "common.h"
#include "BA/mypcl.hpp"
#include "BA/tools.hpp"

class LM_OPTIMIZER
{
public:
	int window_size, jacob_len;
	vector_quad poses, poses_temp;
	vector_vec3d ts, ts_temp;
	std::vector<vector_vec3d*> origin_points;
	std::vector<std::vector<int>*> window_nums;
	
	LM_OPTIMIZER(int win_sz): window_size(win_sz)
	{
		jacob_len = window_size * 6;
		poses.resize(window_size);
		poses_temp.resize(window_size);
		ts.resize(window_size);
		ts_temp.resize(window_size);
	};

  ~LM_OPTIMIZER()
  {
    for(uint i = 0; i < origin_points.size(); i++)
		{
			delete (origin_points[i]);
      delete (window_nums[i]);
		}
		origin_points.clear(); window_nums.clear();
  }

	void get_center(vector_vec3d& origin_pc, int cur_frame, vector_vec3d& origin_point,
                  std::vector<int>& window_num, int filternum2use)
	{
		size_t pt_size = origin_pc.size();
		if(pt_size <= (size_t)filternum2use)
		{
			for(size_t i = 0; i < pt_size; i++)
			{
				origin_point.emplace_back(origin_pc[i]);
				window_num.emplace_back(cur_frame);
			}
			return;
		}

		Eigen::Vector3d center;
		double part = 1.0 * pt_size / filternum2use;

		for(int i = 0; i < filternum2use; i++)
		{
			size_t np = part * i;
			size_t nn = part * (i + 1);
			center.setZero();
			for(size_t j = np; j < nn; j++)
				center += origin_pc[j];

			center = center / (nn - np);
			origin_point.emplace_back(center);
			window_num.emplace_back(cur_frame);
		}
	}

	void push_voxel(std::vector<vector_vec3d*>& origin_pc)
	{
		uint points_size = 0;
		for(int i = 0; i < window_size; i++)
			if(!origin_pc[i]->empty())
				points_size++;

		if(points_size <= 1)
			return;
		
		int filternum2use = 4;
		
		vector_vec3d* origin_point = new vector_vec3d();
		std::vector<int>* window_num = new std::vector<int>();
		window_num->reserve(filternum2use * window_size);
		origin_point->reserve(filternum2use * window_size);
		for(int i = 0; i < window_size; i++)
			if(!origin_pc[i]->empty())
				get_center(*origin_pc[i], i, *origin_point, *window_num, filternum2use);

		origin_points.emplace_back(origin_point);
		window_nums.emplace_back(window_num);
	}

	void optimize()
	{
		double u = 0.01, v = 2;
		Eigen::MatrixXd D(jacob_len, jacob_len), Hess(jacob_len, jacob_len);
		Eigen::VectorXd JacT(jacob_len), dxi(jacob_len);

		Eigen::MatrixXd Hess2(jacob_len, jacob_len);
		Eigen::VectorXd JacT2(jacob_len);

		D.setIdentity();
		double residual1, residual2, q;
		bool is_calc_hess = true;

		cv::Mat matA(jacob_len, jacob_len, CV_64F, cv::Scalar::all(0));
		cv::Mat matB(jacob_len, 1, CV_64F, cv::Scalar::all(0));
		cv::Mat matX(jacob_len, 1, CV_64F, cv::Scalar::all(0));

		for(int loop = 0; loop < 20; loop++)
		{
			if(is_calc_hess)
				calculate_HJ(poses, ts, Hess, JacT, residual1);
			
			D = Hess.diagonal().asDiagonal();
			Hess2 = Hess + u * D;

			for(int j = 0; j < jacob_len; j++)
			{
				matB.at<double>(j, 0) = -JacT(j, 0);
				for(int f = 0; f < jacob_len; f++)
					matA.at<double>(j, f) = Hess2(j, f);
			}
			cv::solve(matA, matB, matX, cv::DECOMP_QR);

			for(int j = 0; j < jacob_len; j++)
				dxi(j, 0) = matX.at<double>(j, 0);

			for(int i = 0; i < window_size; i++)
			{
				Eigen::Quaterniond q_tmp(exp(dxi.block<3, 1>(6*i, 0)) * poses[i]);
				Eigen::Vector3d t_tmp(dxi.block<3, 1>(6*i+3, 0) + ts[i]);
				assign_qt(poses_temp[i], ts_temp[i], q_tmp, t_tmp);
			}

			double q1 = 0.5 * (dxi.transpose() * (u * D * dxi - JacT))[0];
			evaluate_only_residual(poses_temp, ts_temp, residual2);

			q = (residual1 - residual2);
			// printf("residual%d: %lf %lf u: %lf v: %lf q: %lf %lf %lf\n",
			// 	   loop, residual1, residual2, u, v, q/q1, q1, q);
			assert(!std::isnan(residual1));
			assert(!std::isnan(residual2));

			if(q > 0)
			{
				for(int i = 0; i < window_size; i++)
					assign_qt(poses[i], ts[i], poses_temp[i], ts_temp[i]);
				q = q / q1;
				v = 2;
				q = 1 - pow(2*q-1, 3);
				u *= (q < 1.0/3 ? 1.0/3:q);
				is_calc_hess = true;
			}
			else
			{
				u = u * v;
				v = 2 * v;
				is_calc_hess = false;
			}

			if(fabs(residual1 - residual2) < 1e-9) break;
		}
	}

	void calculate_HJ(vector_quad& poses, vector_vec3d& ts, Eigen::MatrixXd& Hess,
                    Eigen::VectorXd& JacT, double& residual)
	{
		Hess.setZero();
		JacT.setZero();
		residual = 0;
		Eigen::MatrixXd _hess(Hess);
		Eigen::MatrixXd _jact(JacT);

		size_t voxel_size = origin_points.size();

		for(size_t i = 0; i < voxel_size; i++)
		{
			vector_vec3d& origin_pts = *origin_points[i];
			std::vector<int>& win_num = *window_nums[i];
			size_t pts_size = origin_pts.size();

			Eigen::Vector3d vec_tran;
			vector_vec3d pt_trans(pts_size);
			std::vector<Eigen::Matrix3d> point_xis(pts_size);
			Eigen::Vector3d centor(Eigen::Vector3d::Zero());
			Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

			for(size_t j = 0; j < pts_size; j++)
			{
				vec_tran = poses[win_num[j]] * origin_pts[j];
				point_xis[j] = -wedge(vec_tran);
				pt_trans[j] = vec_tran + ts[win_num[j]];

				centor += pt_trans[j];
				covMat += pt_trans[j] * pt_trans[j].transpose();
			}

			double N = pts_size;
			covMat = covMat - centor * centor.transpose() / N;
			covMat = covMat / N;
			centor = centor / N;

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
			Eigen::Vector3d eigen_value = saes.eigenvalues();

			Eigen::Matrix3d U = saes.eigenvectors();
			Eigen::Vector3d u[3];
			for(int j = 0; j < 3; j++)
				u[j] = U.block<3, 1>(0, j);

			Eigen::Matrix3d ukukT = u[0] * u[0].transpose();
			Eigen::Vector3d vec_Jt;
			for(size_t j = 0; j < pts_size; j++)
			{
				pt_trans[j] = pt_trans[j] - centor;
				vec_Jt = 2.0 / N * ukukT * pt_trans[j];
				_jact.block<3, 1>(6*win_num[j], 0) -= point_xis[j] * vec_Jt;
				_jact.block<3, 1>(6*win_num[j]+3, 0) += vec_Jt;
			}

			Eigen::Matrix3d Hessian33;
			Eigen::Matrix3d F;
			std::vector<Eigen::Matrix3d> F_(3);
			for(size_t j = 0; j < 3; j++)
			{
				if(j == 0)
				{
					F_[j].setZero();
					continue;
				}
				Hessian33 = u[j] * u[0].transpose();
				F_[j] = 1.0 / N / (eigen_value[0] - eigen_value[j]) * 
					(Hessian33 + Hessian33.transpose());
			}

			Eigen::Matrix3d h33;
			size_t rownum, colnum;
			for(size_t j = 0; j < pts_size; j++) // for each point in voxel
			{
				for(int f = 0; f < 3; f++)
					F.block<1, 3>(f, 0) = pt_trans[j].transpose() * F_[f];

				F = U * F;
				colnum = 6 * win_num[j];
				for(size_t k = 0; k < pts_size; k++)
				{
					Hessian33 = u[0] * (pt_trans[k]).transpose() * F + 
						u[0].dot(pt_trans[k]) * F;

					rownum = 6 * win_num[k];
					if(k == j)
						Hessian33 += (N-1) / N * ukukT;
					else
						Hessian33 -= 1.0 / N * ukukT;
					Hessian33 = 2.0 / N * Hessian33;

					_hess.block<3, 3>(rownum+3, colnum+3) += Hessian33;
					h33 = Hessian33 * point_xis[j];
					_hess.block<3, 3>(rownum+3, colnum) += h33;
					_hess.block<3, 3>(rownum, colnum+3) -= point_xis[k] * Hessian33;
					_hess.block<3, 3>(rownum, colnum) -= point_xis[k] * h33;
				}
			}

			residual += eigen_value[0];
			Hess += _hess;
			JacT += _jact;
			_hess.setZero();
			_jact.setZero();
		}
	}

	void evaluate_only_residual(vector_quad& poses_, vector_vec3d& ts_, double& residual)
	{
		residual = 0;
		size_t voxel_size = origin_points.size();
		Eigen::Vector3d pt_trans, new_center;
		Eigen::Matrix3d new_A;

		for(size_t i = 0; i < voxel_size; i++)
		{
			new_center.setZero();
			new_A.setZero();
			for(size_t j = 0; j < window_nums[i]->size(); j++)
			{
				pt_trans = poses_[(*window_nums[i])[j]] * (*origin_points[i])[j] + 
					ts_[(*window_nums[i])[j]];
				new_A += pt_trans * pt_trans.transpose();
				new_center += pt_trans;				
			}
			new_center /= origin_points[i]->size();
			new_A /= origin_points[i]->size();
			new_A -= new_center * new_center.transpose();
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(new_A);
			Eigen::Vector3d eigen_value = saes.eigenvalues();
			residual += eigen_value(0);
		}
	}
};

class OCTO_TREE
{
public:
	std::vector<vector_vec3d*> origin_pc;
	std::vector<vector_vec3d*> transform_pc;
	OCTO_TREE* leaves[8];

	int win_size, eigen_ratio;
	OT_STATE octo_state;
	int points_size, layer;
	
	double voxel_center[3];
	double quater_length;
  Eigen::Vector3d value_vector;

	OCTO_TREE(int window_size, double eigen_limit): win_size(window_size), eigen_ratio(eigen_limit)
	{
		octo_state = UNKNOWN; layer = 0;
		for(int i = 0; i < 8; i++)
			leaves[i] = nullptr;

		for(int i = 0; i < win_size; i++)
		{
			origin_pc.emplace_back(new vector_vec3d());
			transform_pc.emplace_back(new vector_vec3d());
		}
	}

	~OCTO_TREE()
	{
    for(int i = 0; i < win_size; i++)
		{
			delete (origin_pc[i]);
			delete (transform_pc[i]);
		}
		origin_pc.clear();
		transform_pc.clear();
		for(int i = 0; i < 8; i++)
			if(leaves[i] != nullptr)
				delete leaves[i];
	}

	void recut()
	{
		if(octo_state == UNKNOWN)
		{
			points_size = 0;
			for(int i = 0; i < win_size; i++)
				points_size += origin_pc[i]->size();
			
			if(points_size < MIN_PS)
			{
        octo_state = MID_NODE;
				return;
			}

			if(judge_eigen())
      {
        octo_state = PLANE;
        return;
      }
      else
      {
        if(layer == LAYER_LIMIT)
        {
          octo_state = MID_NODE;
          return;
        }

        for(int i = 0; i < win_size; i++)
        {
          uint pt_size = transform_pc[i]->size();
          for(uint j = 0; j < pt_size; j++)
          {
            int xyz[3] = {0, 0, 0};
            for(uint k = 0; k < 3; k++)
              if((*transform_pc[i])[j][k] > voxel_center[k])
                xyz[k] = 1;

            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
            if(leaves[leafnum] == nullptr)
            {
              leaves[leafnum] = new OCTO_TREE(win_size, eigen_ratio);
              leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
              leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
              leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
              leaves[leafnum]->quater_length = quater_length / 2;
              leaves[leafnum]->layer = layer + 1;
            }
            leaves[leafnum]->origin_pc[i]->push_back((*origin_pc[i])[j]);
            leaves[leafnum]->transform_pc[i]->push_back((*transform_pc[i])[j]);
          }
        }
      }
		}

		for(int i = 0; i < 8; i++)
			if(leaves[i] != nullptr)
				leaves[i]->recut();
	}

	bool judge_eigen()
	{
		Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
		Eigen::Vector3d center(0, 0, 0);

		uint pt_size;
		for(int i = 0; i < win_size; i++)
		{
			pt_size = transform_pc[i]->size();
			for(uint j = 0; j < pt_size; j++)
			{
				covMat += (*transform_pc[i])[j] * (*transform_pc[i])[j].transpose();
				center += (*transform_pc[i])[j];
			}
		}
		center /= points_size;
		covMat = covMat / points_size - center * center.transpose();
		/* saes.eigenvalues()[2] is the biggest */
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    value_vector = saes.eigenvalues();
		if(eigen_ratio < saes.eigenvalues()[2] / saes.eigenvalues()[0]) return true;
    return false;
	}

	void tras_display(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud)
	{
    float ref = 255.0*rand()/(RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

		if(octo_state == PLANE)
		{
			for(int i = 0; i < win_size; i++)
				for(uint j = 0; j < transform_pc[i]->size(); j++)
				{
					ap.x = (*transform_pc[i])[j](0);
					ap.y = (*transform_pc[i])[j](1);
					ap.z = (*transform_pc[i])[j](2);
					ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          ap.normal_z = sqrt(value_vector[0]);
					cloud->points.push_back(ap);
				}
		}
		else
		{
      if(layer == LAYER_LIMIT) return;
			layer++;
			for(int i = 0; i < 8; i++)
				if(leaves[i] != nullptr)
					leaves[i]->tras_display(cloud);
		}
	}

	void feed_pt(LM_OPTIMIZER& lm_opt)
	{
		if(octo_state == PLANE)
			lm_opt.push_voxel(origin_pc);
		else
			for(int i = 0; i < 8; i++)
				if(leaves[i] != nullptr)
					leaves[i]->feed_pt(lm_opt);
	}
};

#endif
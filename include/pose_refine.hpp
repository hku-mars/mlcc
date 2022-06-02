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
#include "mypcl.hpp"

#define MIN_PS 5
#define SMALL_EPS 1e-10
#define HASH_P 116101
#define MAX_N 10000000019

class VOXEL_LOC
{
public:
	int64_t x, y, z;

	VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0): x(vx), y(vy), z(vz){}

	bool operator== (const VOXEL_LOC &other) const
	{
		return (x == other.x && y == other.y && z == other.z);
	}
};

namespace std
{
	template<>
	struct hash<VOXEL_LOC>
	{
		size_t operator() (const VOXEL_LOC &s) const
		{
			using std::size_t;
			using std::hash;
			long index_x, index_y, index_z;
			double cub_len = 1.0/8;
			index_x = int(round(floor((s.x)/cub_len + SMALL_EPS)));
			index_y = int(round(floor((s.y)/cub_len + SMALL_EPS)));
			index_z = int(round(floor((s.z)/cub_len + SMALL_EPS)));
			return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
		}
	};
}

struct M_POINT
{
	float xyz[3];
	int count = 0;
};

void downsample_voxel(pcl::PointCloud<PointType>& pc, double voxel_size)
{
	if(voxel_size < 0.01)
		return;

	std::unordered_map<VOXEL_LOC, M_POINT> feature_map;
	size_t pt_size = pc.size();

	for(size_t i = 0; i < pt_size; i++)
	{
		PointType &pt_trans = pc[i];
		float loc_xyz[3];
		for(int j = 0; j < 3; j++)
		{
			loc_xyz[j] = pt_trans.data[j] / voxel_size;
			if(loc_xyz[j] < 0)
				loc_xyz[j] -= 1.0;
		}

		VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
		auto iter = feature_map.find(position);
		if(iter != feature_map.end())
		{
			iter->second.xyz[0] += pt_trans.x;
			iter->second.xyz[1] += pt_trans.y;
			iter->second.xyz[2] += pt_trans.z;
			iter->second.count++;
		}
		else
		{
			M_POINT anp;
			anp.xyz[0] = pt_trans.x;
			anp.xyz[1] = pt_trans.y;
			anp.xyz[2] = pt_trans.z;
			anp.count = 1;
			feature_map[position] = anp;
		}
	}

	pt_size = feature_map.size();
	pc.clear();
	pc.resize(pt_size);

	size_t i = 0;
	for(auto iter = feature_map.begin(); iter != feature_map.end(); ++iter)
	{
		pc[i].x = iter->second.xyz[0] / iter->second.count;
		pc[i].y = iter->second.xyz[1] / iter->second.count;
		pc[i].z = iter->second.xyz[2] / iter->second.count;
		i++;
	}
}

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

class LM_OPTIMIZER
{
public:
	int window_size, jacob_len;
	vector_quad poses, poses_temp;
	vector_vec3d ts, ts_temp;
	std::vector<vector_vec3d*> origin_points;
	std::vector<std::vector<int>*> window_nums;
	
	LM_OPTIMIZER(int win_sz) : window_size(win_sz)
	{
		jacob_len = window_size * 6;
		poses.resize(window_size);
		poses_temp.resize(window_size);
		ts.resize(window_size);
		ts_temp.resize(window_size);
	};

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
		int process_points_size = 0;
		for(int i = 0; i < window_size; i++)
			if(!origin_pc[i]->empty())
				process_points_size++;

		if(process_points_size <= 1)
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

			if(fabs(residual1 - residual2) < 1e-9)
				break;
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

enum OT_STATE {END_OF_TREE, NOT_TREE_END};
class OCTO_TREE
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	static int voxel_windowsize;
	std::vector<vector_vec3d*> origin_pc;
	std::vector<vector_vec3d*> transform_pc;
	OCTO_TREE* leaves[8];

	int capacity, feat_eigen_limit;
	OT_STATE octo_state; // 0 is end of tree, 1 is not
	int points_size, sw_points_size;
	
	double feat_eigen_ratio, feat_eigen_ratio_test;
	double voxel_center[3]; // x, y, z
	double quater_length;
	bool is2opt;

	OCTO_TREE(int capa) : capacity(capa)
	{
		octo_state = END_OF_TREE;
		for(int i = 0; i < 8; i++)
			leaves[i] = nullptr;

		for(int i = 0; i < capacity; i++)
		{
			origin_pc.emplace_back(new vector_vec3d());
			transform_pc.emplace_back(new vector_vec3d());
		}
		is2opt = true;
		feat_eigen_limit = 15;
	}

	virtual ~OCTO_TREE()
	{
		for(int i = 0; i < capacity; i++)
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

	void recut(int layer, size_t frame_head)
	{
		if(octo_state == END_OF_TREE)
		{
			points_size = 0;
			for(int i = 0; i < capacity; i++)
				points_size += origin_pc[i]->size();
			
			if(points_size < MIN_PS)
			{
				feat_eigen_ratio = -1;
				return;
			}

			calc_eigen();
			
			if(std::isnan(feat_eigen_ratio))
			{
				feat_eigen_ratio = -1;
				return;
			}

			if(feat_eigen_ratio >= feat_eigen_limit)
				return;

			if(layer == 4)
				return;

			octo_state = NOT_TREE_END;
		}

		int leafnum;
		size_t pt_size;

		for(int i = frame_head; i < OCTO_TREE::voxel_windowsize; i++)
		{
			pt_size = transform_pc[i]->size();
			for(size_t j = 0; j < pt_size; j++)
			{
				int xyz[3] = {0, 0, 0};
				for(size_t k = 0; k < 3; k++)
				{
					if((*transform_pc[i])[j][k] > voxel_center[k])
						xyz[k] = 1;
				}
				leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
				if(leaves[leafnum] == nullptr)
				{
					leaves[leafnum] = new OCTO_TREE(capacity);
					leaves[leafnum]->voxel_center[0] =
						voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
					leaves[leafnum]->voxel_center[1] =
						voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
					leaves[leafnum]->voxel_center[2] =
						voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
					leaves[leafnum]->quater_length = quater_length / 2;
				}
				leaves[leafnum]->origin_pc[i]->emplace_back((*origin_pc[i])[j]);
				leaves[leafnum]->transform_pc[i]->emplace_back((*transform_pc[i])[j]);
			}
		}

		if(layer != 0)
			for(int i = frame_head; i < OCTO_TREE::voxel_windowsize; i++)
				if(origin_pc[i]->size() != 0)
				{
					vector_vec3d().swap(*origin_pc[i]);
					vector_vec3d().swap(*transform_pc[i]);
				}

		layer++;
		for(size_t i = 0; i < 8; i++)
			if(leaves[i] != nullptr)
				leaves[i]->recut(layer, frame_head);
	}

	void calc_eigen()
	{
		Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
		Eigen::Vector3d center(0, 0, 0);

		size_t pt_size;
		for(int i = 0; i < capacity; i++)
		{
			pt_size = transform_pc[i]->size();
			for(size_t j = 0; j < pt_size; j++)
			{
				covMat += (*transform_pc[i])[j] * (*transform_pc[i])[j].transpose();
				center += (*transform_pc[i])[j];
			}
		}
		center /= points_size;
		covMat = covMat / points_size - center * center.transpose();
		/* saes.eigenvalues()[2] is the biggest */
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
		feat_eigen_ratio = saes.eigenvalues()[2] / saes.eigenvalues()[0];
	}

	void tras_display(pcl::PointCloud<pcl::PointXYZRGB>& pl_feat, int layer = 0)
	{
		if(octo_state != NOT_TREE_END)
		{
			std::vector<unsigned int> colors;
			colors.push_back(static_cast<unsigned int>(rand() % 256));
			colors.push_back(static_cast<unsigned int>(rand() % 256));
			colors.push_back(static_cast<unsigned int>(rand() % 256));
			for(int i = 0; i < capacity; i++)
				for(uint j = 0; j < transform_pc[i]->size(); j++)
				{
					pcl::PointXYZRGB p;
					Eigen::Vector3d pt((*transform_pc[i])[j](0),
									   (*transform_pc[i])[j](1),
									   (*transform_pc[i])[j](2));
					p.x = pt(0);
					p.y = pt(1);
					p.z = pt(2);
					p.b = colors[0];
					p.g = colors[1];
					p.r = colors[2];
					pl_feat.push_back(p);
				}
		}
		else
		{
			layer++;
			for(int i = 0; i < 8; i++)
				if(leaves[i] != nullptr)
					leaves[i]->tras_display(pl_feat, layer);
		}
	}

	void feed_pt(LM_OPTIMIZER& lm_opt)
	{
		if(octo_state == END_OF_TREE)
		{
			sw_points_size = 0;
			for(int i = 0; i < capacity; i++)
				sw_points_size += origin_pc[i]->size();
			
			if(sw_points_size < MIN_PS)
				return;
			
			traversal_opt_calc_eigen();

			if(std::isnan(feat_eigen_ratio_test))
				return;

			if(feat_eigen_ratio_test > feat_eigen_limit)
				lm_opt.push_voxel(origin_pc);
		}
		else
			for(int i = 0; i < 8; i++)
				if(leaves[i] != nullptr)
					leaves[i]->feed_pt(lm_opt);
	}

	void traversal_opt_calc_eigen()
	{
		Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
		Eigen::Vector3d center(0, 0, 0);

		size_t pt_size;
		for(int i = 0; i < capacity; i++)
		{
			pt_size = transform_pc[i]->size();
			for(size_t j = 0; j < pt_size; j++)
			{
				covMat += (*transform_pc[i])[j] * (*transform_pc[i])[j].transpose();
				center += (*transform_pc[i])[j];
			}
		}

		covMat -= center * center.transpose() / sw_points_size; 
		covMat /= sw_points_size;

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
		feat_eigen_ratio_test = (saes.eigenvalues()[2] / saes.eigenvalues()[0]);
	}
};

int OCTO_TREE::voxel_windowsize = 0;

#endif
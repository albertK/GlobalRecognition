#ifndef CRH_POSE_ESTIMATION
#define CRH_POSE_ESTIMATION

#include <pcl/features/crh.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>

#include "pose_estimation/pose_estimation.h"
#include "descriptor_estimation/crh_estimation.h"

template<typename PointT>
class CRHPoseEstimation:public PoseEstimation<PointT>
{
	using PoseEstimation<PointT>::scene_;
	using PoseEstimation<PointT>::input_hypothesis_;
	using PoseEstimation<PointT>::result_hypotheses_;
	
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::CloudVectorType CloudVectorT;
	typedef Hypothesis<PointT> HypothesisT;
	typedef typename std::vector<HypothesisT> HypothesesT;
	
	CRHPoseEstimation();
	virtual void estimate();
	void setMaxResults(int max_results){max_results_ = max_results;}

protected:
	pcl::NormalEstimationOMP<PointT, pcl::Normal> normal_estimation_;
	pcl::CRHEstimation<PointT, pcl::Normal, pcl::Histogram<90> > crh_;
	pcl::CRHAlignment<PointT, 90> crha_;
	int max_results_;
};

template<typename PointT>
CRHPoseEstimation<PointT>::CRHPoseEstimation()
{
	normal_estimation_.setRadiusSearch(0.01f);
	typename pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
	normal_estimation_.setSearchMethod(kdtree);
	
	max_results_ = 5;
}

template<typename PointT>
void CRHPoseEstimation<PointT>::estimate()
{
	PointCloudPtrT model(new PointCloudT);
	pcl::copyPointCloud(input_hypothesis_.cloud, *model);
	
	//estimate normal
	pcl::PointCloud<pcl::Normal>::Ptr scene_normal(new pcl::PointCloud<pcl::Normal>);
	normal_estimation_.setInputCloud(scene_);
	normal_estimation_.compute(*scene_normal);
	
	pcl::PointCloud<pcl::Normal>::Ptr model_normal(new pcl::PointCloud<pcl::Normal>);
	normal_estimation_.setInputCloud(model);
	normal_estimation_.compute(*model_normal);
	
	pcl::PointIndices::Ptr indices(new pcl::PointIndices);
	pcl::removeNaNNormalsFromPointCloud<pcl::Normal>(*scene_normal, *scene_normal, indices->indices);
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(scene_);
	extract.setIndices(indices);
	extract.setNegative(false);
	extract.filter(*scene_);
	
	indices->indices.clear();
	pcl::removeNaNNormalsFromPointCloud<pcl::Normal>(*model_normal, *model_normal, indices->indices);
	extract.setInputCloud(model);
	extract.setIndices(indices);
	extract.setNegative(false);
	extract.filter(*model);
	
	//estimate centroid
	Eigen::Vector4f scene_centroid;
	Eigen::Vector4f model_centroid;
	pcl::compute3DCentroid<PointT>(*scene_, scene_centroid);
	pcl::compute3DCentroid<PointT>(*model, model_centroid);
	
	//estimate camera roll histogram
	pcl::PointCloud<pcl::Histogram<90> >::Ptr scene_crh(new pcl::PointCloud<pcl::Histogram<90> >);
	crh_.setInputCloud(scene_);
	crh_.setInputNormals(scene_normal);
	crh_.setCentroid(scene_centroid);
	crh_.compute(*scene_crh);
	
	
	pcl::PointCloud<pcl::Histogram<90> >::Ptr model_crh(new pcl::PointCloud<pcl::Histogram<90> >);
	crh_.setInputCloud(model);
	crh_.setInputNormals(model_normal);
	crh_.setCentroid(model_centroid);
	crh_.compute(*model_crh);
	
	//align camera roll histogram and get the transform
	crha_.setInputAndTargetView(model, scene_);
	Eigen::Vector3f mc = model_centroid.block(0,0,3,1);
	Eigen::Vector3f sc = scene_centroid.block(0,0,3,1);
	crha_.setInputAndTargetCentroids(mc, sc);
	crha_.align(*model_crh, *scene_crh);
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms;
	crha_.getTransforms(transforms);
	
	result_hypotheses_.clear();
	result_hypotheses_.resize(transforms.size());
	for(unsigned int i = 0; i < transforms.size() && i < max_results_; ++i)
	{
		result_hypotheses_[i].type = input_hypothesis_.type;
		result_hypotheses_[i].fitness = input_hypothesis_.fitness;
		pcl::transformPointCloud<PointT>(input_hypothesis_.cloud, result_hypotheses_[i].cloud, transforms[i]);
		result_hypotheses_[i].pose = transforms[i] * input_hypothesis_.pose;
	}
}

#endif
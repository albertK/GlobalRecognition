#ifndef POST_PROCESSING
#define POST_PROCESSING

#include <algorithm>
#include <boost/filesystem.hpp>
#include <omp.h>

#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "recognition_database/hypothesis.h"

template<typename PointT>
class PostProcessing
{
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::ConstPtr PointCloudConstPtrT;
	typedef Hypothesis<PointT> HypothesisT;
	typedef typename std::vector<HypothesisT> HypothesesT;
	
	PostProcessing();
	void setInputModels(std::string path);
	void setResolution(float leaf);
	void setInputCluster(PointCloudPtrT cluster);
	void setInputScene(PointCloudPtrT scene);
	void setInputHypotheses(HypothesesT& hypotheses);
	void refineHypotheses();
	void getRefinedHypotheses(HypothesesT& hypotheses){hypotheses = hypotheses_;}
	void verifyHypotheses();
	void getGoodHypotheses(HypothesesT& good_hypotheses){good_hypotheses = good_hypotheses_;}
	void getBadHypotheses(HypothesesT& bad_hypotheses){bad_hypotheses = bad_hypotheses_;}
	void verify();
	
protected:
	//typename pcl::IterativeClosestPoint<PointT, PointT> icp_;
	typename pcl::GreedyVerification<PointT, PointT> greedy_hv_;
	typename std::map<std::string, PointCloudT> models_;
	PointCloudPtrT cluster_;
	PointCloudPtrT scene_;
	HypothesesT hypotheses_;
	HypothesesT good_hypotheses_;
	HypothesesT bad_hypotheses_;
	
	float leaf_;
};

template<typename PointT>
PostProcessing<PointT>::PostProcessing():cluster_(new PointCloudT), scene_(new PointCloudT), greedy_hv_(1.0)
{
	leaf_ = 0.005f;
	
	//icp_.setMaxCorrespondenceDistance(2.0*leaf_);
	//icp_.setMaximumIterations(150);
	
	greedy_hv_.setResolution(leaf_);
	greedy_hv_.setInlierThreshold(leaf_);
}

template<typename PointT>
void PostProcessing<PointT>::setResolution(float leaf)
{
    leaf_ = leaf;
    
    greedy_hv_.setResolution(leaf_);
    greedy_hv_.setInlierThreshold(leaf_);
}

template<typename PointT>
void PostProcessing<PointT>::setInputCluster(PointCloudPtrT cluster)
{
	cluster_->clear();
	
	typename pcl::VoxelGrid<PointT> down_;
	down_.setLeafSize (leaf_, leaf_, leaf_);
	down_.setInputCloud(cluster);
	down_.filter(*cluster_);
}

template<typename PointT>
void PostProcessing<PointT>::setInputScene(PointCloudPtrT scene)
{
	scene_->clear();
	typename pcl::VoxelGrid<PointT> down_;
	down_.setLeafSize (leaf_, leaf_, leaf_);
	down_.setInputCloud(scene);
	down_.filter(*scene_);
}

template<typename PointT>
void PostProcessing<PointT>::setInputHypotheses(HypothesesT& hypotheses)
{
	hypotheses_.clear();
	hypotheses_ = hypotheses;
}

template<typename PointT>
void PostProcessing<PointT>::setInputModels(std::string path)
{
	models_.clear();
	for(boost::filesystem::directory_iterator it(path); it != boost::filesystem::directory_iterator(); ++it)
	{
		std::string model_name = it->path().filename().string();
		if(boost::filesystem::exists(it->path()/(model_name+ std::string(".pcd"))))
		{
			PointCloudT model;
			pcl::io::loadPCDFile<PointT>((it->path()/(model_name+ std::string(".pcd"))).string(), model);
			typename pcl::VoxelGrid<PointT> down_;
			down_.setLeafSize (leaf_, leaf_, leaf_);
			down_.setInputCloud(model.makeShared());
			down_.filter(model);
			models_[model_name] = model;
		}
	}
}

template<typename PointT>
void PostProcessing<PointT>::refineHypotheses()
{
	#pragma omp parallel for num_threads(8)
	for(unsigned int i = 0; i < hypotheses_.size(); ++i)
	{
		PointCloudT aligned;
		typename pcl::IterativeClosestPoint<PointT, PointT> icp_;
		icp_.setMaxCorrespondenceDistance(2.0*leaf_);
		icp_.setMaximumIterations(150);
		
		typename pcl::VoxelGrid<PointT> down_;
		down_.setLeafSize (leaf_, leaf_, leaf_);
		down_.setInputCloud(hypotheses_[i].cloud.makeShared());
		down_.filter(hypotheses_[i].cloud);
		
		icp_.setInputSource(hypotheses_[i].cloud.makeShared());
		icp_.setInputTarget(cluster_);
		icp_.align(aligned);
		
		if(icp_.hasConverged())
		{
			hypotheses_[i].cloud = aligned;
			hypotheses_[i].pose = icp_.getFinalTransformation() * hypotheses_[i].pose;
		}
		hypotheses_[i].fitness = icp_.getFitnessScore();//the lower the better
	}
	
	//std::sort(hypotheses_.begin(), hypotheses_.end(), compareHypothesis<PointT>);//INDIN
}

template<typename PointT>
void PostProcessing<PointT>::verifyHypotheses()
{
	good_hypotheses_.clear();
	bad_hypotheses_.clear();
	
	std::vector<PointCloudConstPtrT> transformed_models;
	for(unsigned int i = 0; i < hypotheses_.size(); ++i)
	{
		PointCloudPtrT transformed_model(new PointCloudT);
		pcl::transformPointCloud<PointT>(models_[hypotheses_[i].type], *transformed_model, hypotheses_[i].pose);
		transformed_models.push_back(transformed_model);
	}
	
	greedy_hv_.setSceneCloud(scene_);
	greedy_hv_.setOcclusionCloud(scene_);
	greedy_hv_.addModels(transformed_models, true);
	greedy_hv_.verify();
	std::vector<bool> mask_hv;
	greedy_hv_.getMask(mask_hv);
	std::vector<float> fitness_hv;
	greedy_hv_.getFitnessScores(fitness_hv);
	
	for(unsigned int i = 0; i < mask_hv.size(); ++i)
	{
		hypotheses_[i].cloud = (*transformed_models[i]);
		hypotheses_[i].fitness = fitness_hv[i];
		if(mask_hv[i] == true)
			good_hypotheses_.push_back(hypotheses_[i]);
		else
			bad_hypotheses_.push_back(hypotheses_[i]);
	}
	std::sort(good_hypotheses_.begin(), good_hypotheses_.end(), compareHypothesis<PointT>);//INDIN
}

template<typename PointT>
void PostProcessing<PointT>::verify()
{
	refineHypotheses();
	verifyHypotheses();
}

#endif
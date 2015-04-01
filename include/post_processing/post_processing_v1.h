#ifndef POST_PROCESSING
#define POST_PROCESSING

#include <algorithm>
#include <boost/filesystem.hpp>

#include <pcl/registration/icp.h>
#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include "recognition_database/hypothesis.h"

template<typename PointT>
class PostProcessing
{
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::ConstPtr PointCloudConstPtrT;
	typedef typename PointCloudT::CloudVectorType CloudVectorT;
	typedef typename std::vector<std::vector<Hypothesis<PointT> > > Hypotheses;
	
	PostProcessing();
	void setInputModels(std::string path);
	void setResolution(float leaf){leaf_ = leaf;}
	void setInputClusters(CloudVectorT& clusters);
	void setInputHypotheses(Hypotheses& hypotheses);
	void refineHypotheses();
	void getRefinedHypotheses(Hypotheses& hypotheses){hypotheses = hypotheses_;}
	void verifyHypotheses();
	void sortVerifiedGoodHypotheses();
	void getGoodHypotheses(Hypotheses& good_hypotheses){good_hypotheses = good_hypotheses_;}
	void getBadHypotheses(Hypotheses& bad_hypotheses){bad_hypotheses = bad_hypotheses_;}
	void verify();
	
protected:
	typename pcl::IterativeClosestPoint<PointT, PointT> icp_;
	typename pcl::GreedyVerification<PointT, PointT> greedy_hv_;
	typename std::map<std::string, PointCloudT> models_;
	CloudVectorT clusters_;
	Hypotheses hypotheses_;
	Hypotheses good_hypotheses_;
	Hypotheses bad_hypotheses_;
	
	float leaf_;
};

template<typename PointT>
PostProcessing<PointT>::PostProcessing()
{
	leaf_ = 0.005f;
	
	icp_.setMaxCorrespondenceDistance(2.0f*leaf_);
	icp_.setMaximumIterations(100);
	
	greedy_hv_.setResolution(leaf_);
	greedy_hv_.setInlierThreshold(1.5f*leaf_);
}

template<typename PointT>
void PostProcessing<PointT>::setInputClusters(CloudVectorT& clusters)
{
	clusters_.clear();
	clusters_ = clusters;
}

template<typename PointT>
void PostProcessing<PointT>::setInputHypotheses(Hypotheses& hypotheses)
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
			models_[model_name] = model;
		}
	}
}

template<typename PointT>
void PostProcessing<PointT>::refineHypotheses()
{
	for(unsigned int i = 0; i < hypotheses_.size(); ++i)
	{
		for(unsigned int j = 0; j < hypotheses_[i].size(); ++j)
		{
			PointCloudT aligned;
			icp_.setInputSource(hypotheses_[i][j].cloud.makeShared());
			icp_.setInputTarget(clusters_[i].makeShared());
			icp_.align(aligned);
			
			if(icp_.hasConverged())
			{
				hypotheses_[i][j].cloud = aligned;
				hypotheses_[i][j].pose = icp_.getFinalTransformation() * hypotheses_[i][j].pose;
				
			}
			
			hypotheses_[i][j].fitness = icp_.getFitnessScore();//the lower the better
		}
	}
}

template<typename PointT>
void PostProcessing<PointT>::verifyHypotheses()
{
	good_hypotheses_.clear();
	good_hypotheses_.resize(clusters_.size());
	bad_hypotheses_.clear();
	bad_hypotheses_.resize(clusters_.size());
	
	for(unsigned int i = 0; i < hypotheses_.size(); ++i)
	{
		std::vector<PointCloudConstPtrT> transformed_models;
		for(unsigned int j = 0; j < hypotheses_[i].size(); ++j)
		{
			PointCloudPtrT transformed_model(new PointCloudT);
			pcl::transformPointCloud<PointT>(models_[hypotheses_[i][j].type], *transformed_model, hypotheses_[i][j].pose);
			transformed_models.push_back(transformed_model);
		}
		
		greedy_hv_.setSceneCloud(clusters_[i].makeShared());
		greedy_hv_.addModels(transformed_models, true);
		greedy_hv_.verify();
		std::vector<bool> mask_hv;
		greedy_hv_.getMask(mask_hv);
		
		for(unsigned int j = 0; j < mask_hv.size(); ++j)
		{
			if(mask_hv[j] == true)
			{
				good_hypotheses_[i].push_back(hypotheses_[i][j]);
			}
			else
			{
				bad_hypotheses_[i].push_back(hypotheses_[i][j]);
			}
		}
	}
}

template<typename PointT>
void PostProcessing<PointT>::sortVerifiedGoodHypotheses()
{
	for(unsigned int i = 0; i < good_hypotheses_.size(); ++i)
		std::sort(good_hypotheses_[i].begin(), good_hypotheses_[i].end(), compareHypothesis<PointT>);
}

template<typename PointT>
void PostProcessing<PointT>::verify()
{
	refineHypotheses();
	verifyHypotheses();
	sortVerifiedGoodHypotheses();
}

#endif
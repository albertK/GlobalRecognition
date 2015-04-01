#ifndef POSE_ESTIMATION
#define POSE_ESTIMATION

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include "recognition_database/hypothesis.h"

template<typename PointT>
class PoseEstimation
{
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::CloudVectorType CloudVectorT;
	typedef Hypothesis<PointT> HypothesisT;
	typedef typename std::vector<HypothesisT> HypothesesT;
	
	PoseEstimation():scene_(new PointCloudT){}
	void setInputScene(PointCloudPtrT scene);
	void setInputHypothesis(HypothesisT& hypothesis){input_hypothesis_ = hypothesis;}
	virtual void estimate() = 0;
	void getResultHypotheses(HypothesesT& hypotheses){hypotheses = result_hypotheses_;}
	
protected:
	PointCloudPtrT scene_;
	HypothesisT input_hypothesis_;
	HypothesesT result_hypotheses_;
};

template<typename PointT>
void PoseEstimation<PointT>::setInputScene(PointCloudPtrT scene)
{
	scene_->clear();
	pcl::copyPointCloud(*scene, *scene_);
}

#endif
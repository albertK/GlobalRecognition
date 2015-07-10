#ifndef GLOBAL_RECOGNITION
#define GLOBAL_RECOGNITION

#include <typeinfo>
#include <boost/shared_ptr.hpp>

#include <pcl/common/time.h>
#include <pcl/apps/3d_rec_framework/utils/metrics.h>

#include "pre_processing/pre_processing.h"

#include "descriptor_estimation/vfh_estimation.h"
#include "descriptor_estimation/cvfh_estimation.h"
#include "descriptor_estimation/ourcvfh_estimation.h"
#include "descriptor_estimation/esf_estimation.h"

#include "recognition_database/recognition_database.h"

#include "pose_estimation/crh_pose_estimation.h"

#include "post_processing/post_processing.h"

template<typename PointT>
class GlobalRecognition
{
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::ConstPtr PointCloudConstPtrT;
	typedef typename PointCloudT::CloudVectorType CloudVectorT;
	typedef Hypothesis<PointT> HypothesisT;
	typedef typename std::vector<std::vector<HypothesisT> > Hypotheses;
	
	GlobalRecognition():scene_(new PointCloudT),k_(9){}
	~GlobalRecognition(){}
	bool init(std::string db_path, std::string roi);
	void setInputScene(const PointCloudPtrT scene);
	void setNumCandidates(int k){k_ = k;}
	void recognize();
	void getClusters(CloudVectorT& clusters){clusters = clusters_;}
	void getRecognitionResult(Hypotheses& hypotheses){hypotheses = hypotheses_;}
	
protected:
	PreProcessing<PointT> pre_processing_;
	
	/*different kind of descriptors
	 
	 VFH:
	 VFHEstimation<PointT, pcl::VFHSignature308>
	 
	 CVFH:
	 CVFHEstimation<PointT, pcl::VFHSignature308>
	 
	 OUR-CVFH:
	 OURCVFHEstimation<PointT, pcl::VFHSignature308>
	 
	 ESF:
	 ESFEstimation<PointT, pcl::ESFSignature640>
	 
	 */
	ESFEstimation<PointT, pcl::ESFSignature640> descriptor_estimation_;
	
	/*different kind of descriptors
	 * 
	 VFH:
	 RecognitionDatabase<PointT, pcl::VFHSignature308, flann::L1<float> >
	 
	 CVFH:
	 RecognitionDatabase<PointT, pcl::VFHSignature308, Metrics::HistIntersectionUnionDistance<float> >
	 
	 OUR-CVFH:
	 RecognitionDatabase<PointT, pcl::VFHSignature308, Metrics::HistIntersectionUnionDistance<float> >
	 
	 ESF:
	 RecognitionDatabase<PointT, pcl::ESFSignature640, flann::L1<float> >
	 
	 */
	RecognitionDatabase<PointT, pcl::ESFSignature640, flann::L1<float> > recognition_database_;
	
	CRHPoseEstimation<PointT> pose_estimation_;
	PostProcessing<PointT> post_processing_;
	
	PointCloudPtrT scene_;
	CloudVectorT clusters_;
	
	/*different kind of descriptors
	 
	 ESF:
	 pcl::PointCloud<pcl::ESFSignature640>
	 
	 VFH, CVFH, OUR-CVFH:
	 pcl::PointCloud<pcl::VFHSignature308>
	 
	*/
	pcl::PointCloud<pcl::ESFSignature640> descriptors_;
	
	Hypotheses hypotheses_;
	int k_;
};

template<typename PointT>
bool GlobalRecognition<PointT>::init(std::string db_path, std::string roi)
{
	
	if(!pre_processing_.setROIBoundary(roi))
	{
		pcl::console::print_error("Initialization failed, wrong ROI...\n");
		return false;
	}
	
	if(typeid(descriptor_estimation_) == typeid(ESFEstimation<PointT, pcl::ESFSignature640>))
	{
		pcl::console::print_highlight("Use ESF descriptor\n");
		recognition_database_.setDescriptor("esf");
	}
	else if(typeid(descriptor_estimation_) == typeid(VFHEstimation<PointT, pcl::VFHSignature308>))
	{
		pcl::console::print_highlight("Use VFH descriptor\n");
		recognition_database_.setDescriptor("vfh");
	}
	else if(typeid(descriptor_estimation_) == typeid(CVFHEstimation<PointT, pcl::VFHSignature308>))
	{
		pcl::console::print_highlight("Use CVFH descriptor\n");
		recognition_database_.setDescriptor("cvfh");
	}
	else if(typeid(descriptor_estimation_) == typeid(OURCVFHEstimation<PointT, pcl::VFHSignature308>))
	{
		pcl::console::print_highlight("Use OURCVFH descriptor\n");
		recognition_database_.setDescriptor("ourcvfh");
	}
	
	pre_processing_.setLeafSize(0.003f);
	pre_processing_.setClusterSize(500,10000);
	pre_processing_.setClusterTolerance(0.05f);
	
	recognition_database_.loadDB(db_path);
	
	pose_estimation_.setMaxResults(3);
	
	post_processing_.setResolution(0.005f);
	post_processing_.setInputModels(db_path);
	
	return true;
}

template<typename PointT>
void GlobalRecognition<PointT>::setInputScene(const PointCloudPtrT scene)
{
	scene_->clear();
	pcl::copyPointCloud(*scene, *scene_);
}

template<typename PointT>
void GlobalRecognition<PointT>::recognize()
{
	hypotheses_.clear();
	clusters_.clear();
	
	{
		pcl::ScopeTime frame_process ("Scene Pre-processing -------------");
		pre_processing_.setInputScene(scene_);
		pre_processing_.process();
		pre_processing_.getResultClusters(clusters_);
	}
	if(clusters_.size() == 0)
	{
		pcl::console::print_warn("No clusters detected...\n");
		return;
	}
	else
	{
		pcl::console::print_highlight("%d clusters detected\n", clusters_.size());
		hypotheses_.resize(clusters_.size());
	}
	
	Hypotheses raw_hypotheses;
	{
		pcl::ScopeTime frame_process ("Type Recognition -------------");
		
		descriptor_estimation_.setInputClusters(clusters_);
		descriptor_estimation_.estimate();
		descriptor_estimation_.getResultDescriptors(descriptors_);
		
		recognition_database_.queryDB(descriptors_.makeShared(), k_, raw_hypotheses);
	}
	
	Hypotheses pose_hypotheses;
	pose_hypotheses.resize(clusters_.size());
	for(unsigned int i = 0; i < raw_hypotheses.size(); ++i)
	{
		{
			pcl::ScopeTime frame_process ("Pose Estimation -------------");
			pose_estimation_.setInputScene(clusters_[i].makeShared());
			for(unsigned int j = 0; j < raw_hypotheses[i].size(); ++j)
			{
				pose_estimation_.setInputHypothesis(raw_hypotheses[i][j]);
				pose_estimation_.estimate();
				std::vector<HypothesisT> vec_hypothesis;
				pose_estimation_.getResultHypotheses(vec_hypothesis);
				for(unsigned int k = 0; k < vec_hypothesis.size(); ++k)
				{
					pose_hypotheses[i].push_back(vec_hypothesis[k]);
				}
			}
		}
		
		post_processing_.setInputCluster(clusters_[i].makeShared());
		post_processing_.setInputScene(scene_);
		post_processing_.setInputHypotheses(pose_hypotheses[i]);
		{
			pcl::ScopeTime frame_process ("ICP -------------");
			post_processing_.refineHypotheses();
		}
		{
			pcl::ScopeTime frame_process ("HV -------------");
			post_processing_.verifyHypotheses();
		}
		post_processing_.getGoodHypotheses(hypotheses_[i]);
	}
}

#endif
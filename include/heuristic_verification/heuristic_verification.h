#ifndef HEURISTIC_VERIFICATION
#define HEURISTIC_VERIFICATION

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

#include <pcl/recognition/hv/hypotheses_verification.h>

namespace pcl
{
	template<typename ModelT, typename SceneT>
	class HeuristicVerification : public HypothesisVerification<ModelT, SceneT>
	{
		using HypothesisVerification<ModelT, SceneT>::mask_;
		using HypothesisVerification<ModelT, SceneT>::scene_cloud_;
		using HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
		using HypothesisVerification<ModelT, SceneT>::scene_downsampled_tree_;
		using HypothesisVerification<ModelT, SceneT>::visible_models_;
		using HypothesisVerification<ModelT, SceneT>::resolution_;
		using HypothesisVerification<ModelT, SceneT>::inliers_threshold_;
		
	protected:
		std::vector<float> fitness_scores_;
		float pass_thres_;
		typename std::vector<typename pcl::PointCloud<ModelT>::Ptr> visible_models_downsampled_;
		typename std::vector<typename pcl::search::KdTree<ModelT>::Ptr> visible_models_downsampled_trees_;
		
	public:
		HeuristicVerification():pass_thres_(0.9){}
		void setPassThres(float pass_thres){pass_thres_ = pass_thres;}
		void init();
		virtual void verify();
		void getFitnessScores(std::vector<float>& fitness_scores){fitness_scores = fitness_scores_;}
	};
};

template<typename ModelT, typename SceneT>
void pcl::HeuristicVerification<ModelT, SceneT>::init()
{
	//init mask_ and fitness_scores_
	fitness_scores_.resize(visible_models_.size());
	mask_.resize(visible_models_.size());
	for(unsigned int i = 0; i < visible_models_.size(); ++i)
	{
		fitness_scores_.at(i) = 0.0;
		mask_[i] = false;
	}
	
	//downsample visible_models_
	visible_models_downsampled_.clear();
	visible_models_downsampled_.resize(visible_models_.size());
	for(unsigned int i = 0; i < visible_models_.size(); ++i)
	{
		visible_models_downsampled_.at(i).reset(new pcl::PointCloud<ModelT>());
		
		typename pcl::VoxelGrid<SceneT> down_;
		down_.setLeafSize(resolution_, resolution_, resolution_);
		down_.setInputCloud(visible_models_.at(i));
		down_.filter(*visible_models_downsampled_.at(i));
	}
	
	//build kdtree for visible_models_
	visible_models_downsampled_trees_.resize(visible_models_downsampled_.size());
	for(unsigned int i = 0; i < visible_models_downsampled_.size(); ++i)
	{
		visible_models_downsampled_trees_.at(i).reset(new pcl::search::KdTree<ModelT>);
		if(visible_models_downsampled_.at(i)->size() != 0)
			visible_models_downsampled_trees_.at(i)->setInputCloud(visible_models_downsampled_.at(i));
	}
}

template<typename ModelT, typename SceneT>
void pcl::HeuristicVerification<ModelT, SceneT>::verify()
{
	init();
	
	for(unsigned int i = 0; i < visible_models_downsampled_trees_.size(); ++i)
	{
		int num_inliers = 0;
		std::vector<int> nn_indices;
		std::vector<float> nn_distances;
		if(visible_models_downsampled_.at(i)->size() != 0)
		{
			for(unsigned int j = 0; j < scene_cloud_downsampled_->size(); ++j)
			{
				if(visible_models_downsampled_trees_.at(i)->radiusSearch(scene_cloud_downsampled_->points[j], inliers_threshold_, nn_indices, nn_distances))
					num_inliers++;
			}
		}
		float score_explain_cluster = ((float) num_inliers) / ((float) scene_cloud_downsampled_->size());
		
		num_inliers = 0;
		for(unsigned int j = 0; j < visible_models_downsampled_.at(i)->size(); ++j)
		{
			if(scene_downsampled_tree_->radiusSearch(visible_models_downsampled_.at(i)->points[j], inliers_threshold_, nn_indices, nn_distances))
				num_inliers++;
		}
		float score_explain_model = ((float) num_inliers) / ((float) visible_models_downsampled_.at(i)->size());
		
		fitness_scores_.at(i) = std::min(score_explain_cluster, score_explain_model);
		if(fitness_scores_.at(i) > pass_thres_)
			mask_.at(i) = true;
	}
}

#endif
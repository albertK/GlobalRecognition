#ifndef RECOGNITION_DATABASE
#define RECOGNITION_DATABASE

#include <string>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/common/transforms.h>

#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkPolyData.h>

#include "recognition_database/hypothesis.h"
#include "descriptor_estimation/vfh_estimation.h"
#include "descriptor_estimation/cvfh_estimation.h"
#include "descriptor_estimation/ourcvfh_estimation.h"
#include "descriptor_estimation/esf_estimation.h"

template<typename PointT, typename DescriptorT, typename DistT>
class RecognitionDatabase
{
public:
	typedef typename pcl::PointCloud<PointT> PointCloudT;
	typedef typename PointCloudT::Ptr PointCloudPtrT;
	typedef typename PointCloudT::CloudVectorType CloudVectorT;
	typedef typename pcl::PointCloud<DescriptorT> DescriptorCloudT;
	typedef typename DescriptorCloudT::Ptr DescriptorCloudPtrT;
	typedef typename std::vector<std::vector<Hypothesis<PointT> > > Hypotheses;
	typedef typename pcl::search::FlannSearch<DescriptorT, DistT> SearchT;
	
	RecognitionDatabase();
	RecognitionDatabase(int trees, int checks);
	void setDescriptor(std::string descriptor){descriptor_name_ = descriptor;}
	void trainDB(std::string model_dir, std::string ouput_dir, bool force_retrain=false, int tesselation_level=1, float leaf=0.005f);
	void loadDB(std::string db_dir);
	void queryDB(DescriptorCloudPtrT query, int k, Hypotheses& clusters_hypotheses);
	
protected:
	void writeFloat(std::string path, float num);
	void writeMatrix4f(std::string path, Eigen::Matrix4f mat);
	void readFloat(std::string path, float& num);
	void readMatrix4f(std::string path, Eigen::Matrix4f& mat);
	std::string descriptor_name_;
	SearchT flann_search_;
	DescriptorCloudPtrT descriptors_;
	std::vector<std::string> names_;
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;
	CloudVectorT views_;
};

template<typename PointT, typename DescriptorT, typename DistT>
RecognitionDatabase<PointT, DescriptorT, DistT>::RecognitionDatabase():flann_search_(true, typename SearchT::FlannIndexCreatorPtr(new typename SearchT::KdTreeMultiIndexCreator(4))),descriptors_(new DescriptorCloudT)
{
	flann_search_.setPointRepresentation(typename SearchT::PointRepresentationPtr (new pcl::DefaultFeatureRepresentation<DescriptorT>));
	flann_search_.setChecks(256);
}

template<typename PointT, typename DescriptorT, typename DistT>
RecognitionDatabase<PointT, DescriptorT, DistT>::RecognitionDatabase(int trees, int checks):flann_search_(true, typename SearchT::FlannIndexCreatorPtr(new typename SearchT::KdTreeMultiIndexCreator(trees))),descriptors_(new DescriptorCloudT)
{
	flann_search_.setPointRepresentation(typename SearchT::PointRepresentationPtr (new pcl::DefaultFeatureRepresentation<DescriptorT>));
	flann_search_.setChecks(checks);
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::loadDB(std::string db_dir)
{
	descriptors_->clear();
	names_.clear();
	poses_.clear();
	views_.clear();
	
	for(boost::filesystem::directory_iterator it(db_dir); it != boost::filesystem::directory_iterator(); ++it)
	{
		for(boost::filesystem::directory_iterator descriptor_it((it->path()/"descriptors")/descriptor_name_); descriptor_it != boost::filesystem::directory_iterator(); ++descriptor_it)
		{
			std::string file_num = descriptor_it->path().filename().replace_extension().string().substr(11);
			
			//load model descriptor
			DescriptorCloudPtrT descriptor(new DescriptorCloudT);
			pcl::io::loadPCDFile<DescriptorT>(descriptor_it->path().string(), *descriptor);
			descriptors_->push_back(descriptor->at(0));
			
			//load model name
			names_.push_back(it->path().filename().string());
			
			//load model pose
			std::string pose_file_name = std::string("pose_") + file_num + std::string(".txt");
			Eigen::Matrix4f pose;
			readMatrix4f(((it->path()/"poses")/pose_file_name).string(), pose);
			poses_.push_back(pose);
			
			//load each view
			std::string view_file_name = std::string("view_") + file_num + std::string(".pcd");
			PointCloudT view;
			pcl::io::loadPCDFile<PointT>(((it->path()/"views")/view_file_name).string(), view);
			views_.push_back(view);
		}
	}
	flann_search_.setInputCloud(descriptors_);
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::trainDB(std::string model_dir, std::string output_dir, bool force_retrain, int tesselation_level, float leaf)
{
	pcl::visualization::PCLVisualizer renderer("render");
	
	boost::filesystem::path model_path(model_dir);
	boost::filesystem::path output_path(output_dir);
	
	//iterate through all the model
	for(boost::filesystem::directory_iterator it(model_path); it!=boost::filesystem::directory_iterator(); ++it)
	{
		if(it->path().filename().extension().string() != ".ply")
			continue;
		
		std::string model_name = it->path().filename().replace_extension().string();
		
		//check if the model has been trained
		if(boost::filesystem::exists(output_path/model_name))
		{
			if(force_retrain)
				boost::filesystem::remove_all(output_path/model_name);
			else
			{
				pcl::console::print_highlight("Model: %s has been trained\n", model_name.c_str());
				continue;
			}
		}
		
		pcl::console::print_highlight("Rendering model:%s\n", model_name.c_str());
		
		//create some directories to store the trained models
		boost::filesystem::create_directory(output_path/model_name);
		boost::filesystem::create_directory(output_path/model_name/"views");
		boost::filesystem::create_directory(output_path/model_name/"poses");
		boost::filesystem::create_directory(output_path/model_name/"enthropies");
		boost::filesystem::create_directory(output_path/model_name/"descriptors");
		boost::filesystem::copy_file(it->path(), output_path/model_name/(model_name+".ply"));
		
		//read in the ply model
		vtkSmartPointer<vtkPLYReader> readerQuery = vtkSmartPointer<vtkPLYReader>::New();
		readerQuery->SetFileName (it->path().c_str());
		vtkSmartPointer<vtkPolyData> polydata = readerQuery->GetOutput();
		polydata->Update();
		
		/*render the ply model from different view*/
		//add the model to PCLVisualizer
		renderer.addModelFromPolyData(polydata, "model", 0);
		//input parameter
		const int xres = 200;
		const int yres = 200;
		const float view_angle = 60;
		//output parameter
		pcl::PointCloud<pcl::PointXYZ>::CloudVectorType views;
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
		std::vector<float> enthropies;
		//do the rendering
		renderer.renderViewTesselatedSphere(xres, yres, views, poses, enthropies, tesselation_level, view_angle);
		renderer.removeCorrespondences("model");
		renderer.close();
		
		PointCloudT completeModel;
		
		VFHEstimation<pcl::PointXYZ> vfh_estimation;
		CVFHEstimation<pcl::PointXYZ> cvfh_estimation;
		OURCVFHEstimation<pcl::PointXYZ> ourcvfh_estimation;
		ESFEstimation<pcl::PointXYZ> esf_estimation;
		//iterate through all the views
		for(int i = 0; i < views.size(); ++i)
		{
			//add the partial view to the complete model
			Eigen::Matrix4f pose_inverse = Eigen::Matrix4f::Identity();
			pose_inverse.block(0,0,3,3) = poses[i].block(0,0,3,3).transpose();
			pose_inverse.block(0,3,3,1) = - poses[i].block(0,0,3,3).transpose() * poses[i].block(0,3,3,1);
			PointCloudT transformed_view;
			pcl::transformPointCloud<pcl::PointXYZ>(views[i], transformed_view, pose_inverse);
			completeModel += transformed_view;
			
			/*save the view, pose, enthropy, and descriptor to the disk*/
			//down sample the current view
			pcl::VoxelGrid<pcl::PointXYZ> down;
			down.setLeafSize (leaf, leaf, leaf);
			down.setInputCloud (views[i].makeShared());
			down.filter (views[i]);
			
			//save the view to a pcd file
			std::stringstream view_name;
			view_name<<(output_path/model_name/"views/").string()<<"view_"<<i<<".pcd";
			pcl::io::savePCDFileBinary<pcl::PointXYZ>(view_name.str(), views[i]);
			
			//save the transform of each view to a txt file
			std::stringstream pose_name;
			pose_name<<(output_path/model_name/"poses/").string()<<"pose_"<<i<<".txt";
			writeMatrix4f(pose_name.str(), poses[i]);
			
			//save the enthropy of each view to a txt file
			std::stringstream enthropy_name;
			enthropy_name<<(output_path/model_name/"enthropies/").string()<<"enthropy_"<<i<<".txt";
			writeFloat(enthropy_name.str(), enthropies[i]);
			
			//save the descriptor of each view
			//vfh
			vfh_estimation.setInputCluster(views[i]);
			vfh_estimation.estimate();
			pcl::PointCloud<pcl::VFHSignature308> vfhs;
			vfh_estimation.getResultDescriptors(vfhs);
			
			boost::filesystem::create_directory(output_path/model_name/"descriptors/vfh");
			std::stringstream descriptor_name;
			descriptor_name<<(output_path/model_name/"descriptors/vfh/").string()<<"descriptor_"<<i<<".pcd";
			pcl::io::savePCDFileBinary<pcl::VFHSignature308>(descriptor_name.str(), vfhs);
			//cvfh
			cvfh_estimation.setInputCluster(views[i]);
			cvfh_estimation.estimate();
			pcl::PointCloud<pcl::VFHSignature308> cvfhs;
			cvfh_estimation.getResultDescriptors(cvfhs);
			
			boost::filesystem::create_directory(output_path/model_name/"descriptors/cvfh");
			descriptor_name.str("");
			descriptor_name<<(output_path/model_name/"descriptors/cvfh/").string()<<"descriptor_"<<i<<".pcd";
			pcl::io::savePCDFileBinary<pcl::VFHSignature308>(descriptor_name.str(), cvfhs);
			//ourcvfh
			ourcvfh_estimation.setInputCluster(views[i]);
			ourcvfh_estimation.estimate();
			pcl::PointCloud<pcl::VFHSignature308> ourcvfhs;
			ourcvfh_estimation.getResultDescriptors(ourcvfhs);
			
			boost::filesystem::create_directory(output_path/model_name/"descriptors/ourcvfh");
			descriptor_name.str("");
			descriptor_name<<(output_path/model_name/"descriptors/ourcvfh/").string()<<"descriptor_"<<i<<".pcd";
			pcl::io::savePCDFileBinary<pcl::VFHSignature308>(descriptor_name.str(), ourcvfhs);
			//esf
			esf_estimation.setInputCluster(views[i]);
			esf_estimation.estimate();
			pcl::PointCloud<pcl::ESFSignature640> esfs;
			esf_estimation.getResultDescriptors(esfs);
			
			boost::filesystem::create_directory(output_path/model_name/"descriptors/esf");
			descriptor_name.str("");
			descriptor_name<<(output_path/model_name/"descriptors/esf/").string()<<"descriptor_"<<i<<".pcd";
			pcl::io::savePCDFileBinary<pcl::ESFSignature640>(descriptor_name.str(), esfs);
		}
		
		//save the complete model to a pcd file
		pcl::VoxelGrid<pcl::PointXYZ> down;
		down.setLeafSize(leaf, leaf, leaf);
		down.setInputCloud(completeModel.makeShared());
		down.filter(completeModel);
		std::stringstream complete_model_name;
		complete_model_name<<(output_path/model_name).string()<<'/'<<model_name<<".pcd";
		pcl::io::savePCDFileBinary<pcl::PointXYZ>(complete_model_name.str(), completeModel);
	}
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::queryDB(DescriptorCloudPtrT query, int k, Hypotheses& clusters_hypotheses)
{
	clusters_hypotheses.clear();
	clusters_hypotheses.resize(query->size());
	for(unsigned int i = 0; i < clusters_hypotheses.size(); ++i)
		clusters_hypotheses[i].resize(k);
	
	std::vector<std::vector<int> > matched_k_indices;
	std::vector<std::vector<float> > matched_k_sqr_distances;
	flann_search_.nearestKSearch(*query, std::vector<int>(), k, matched_k_indices, matched_k_sqr_distances);
	
	for(unsigned int i = 0; i < clusters_hypotheses.size(); ++i)
	{
		for(unsigned int j = 0; j < k; ++j)
		{
			int index = matched_k_indices[i][j];
			clusters_hypotheses[i][j].type = names_[index];
			clusters_hypotheses[i][j].pose = poses_[index];
			clusters_hypotheses[i][j].fitness = matched_k_sqr_distances[i][index];
			clusters_hypotheses[i][j].cloud = views_[index];
		}
	}
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::writeFloat(std::string path, float num)
{
	std::ofstream fout(path.c_str());
	fout<<num;
	fout.close();
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::writeMatrix4f(std::string path, Eigen::Matrix4f mat)
{
	std::ofstream fout(path.c_str());
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < 3; j++)
			fout<<mat(i,j)<<' ';
		fout<<mat(i,3)<<std::endl;
	}
	fout.close();
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::readFloat(std::string path, float& num)
{
	std::ifstream fin(path.c_str());
	fin>>num;
	fin.close();
}

template<typename PointT, typename DescriptorT, typename DistT>
void RecognitionDatabase<PointT, DescriptorT, DistT>::readMatrix4f(std::string path, Eigen::Matrix4f& mat)
{
	std::ifstream fin(path.c_str());
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 4; j++)
			fin>>mat(i,j);
		
	fin.close();
}

#endif
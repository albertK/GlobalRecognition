#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>

#include "post_processing/post_processing.h"

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	//load test cloud
	std::string test;
	if(pcl::console::parse_argument(argc, argv, "-test", test) == -1)
	{
		std::cerr<<"Please provide the test cloud"<<std::endl;
		return -1;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>(test, *cloud);
	
	//test cloud type
	std::string type;
	if(pcl::console::parse_argument(argc, argv, "-type", type) == -1)
	{
		std::cerr<<"Please provide the type of the test cloud"<<std::endl;
		return -1;
	}
	
	//load pose of the test cloud
	std::string pose;
	if(pcl::console::parse_argument(argc, argv, "-pose", pose) == -1)
	{
		std::cerr<<"Please provide the pose of the test cloud"<<std::endl;
		return -1;
	}
	std::ifstream fin(pose.c_str());
	Eigen::Matrix4f cloud_pose;
	for(unsigned int i = 0; i < 4; ++i)
		for(unsigned int j = 0; j < 4; ++j)
			fin>>cloud_pose(i,j);
		
	cloud_pose = cloud_pose.inverse();
	pcl::transformPointCloud<pcl::PointXYZ>(*cloud, *cloud, cloud_pose);
	
	pcl::visualization::PCLVisualizer visualizer("Result");
	visualizer.setBackgroundColor (0.0, 0.0, 0.0);
	visualizer.addCoordinateSystem (0.5);
	visualizer.registerKeyboardCallback(keyboardEventOccurred, (void*)&visualizer);
	visualizer.addPointCloud<pcl::PointXYZ>(cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud, 0.0, 255.0, 0.0), "scene");
	visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "scene");
	
	//load model pcd
	std::string models;
	if(pcl::console::parse_argument(argc, argv, "-models", models) == -1)
	{
		std::cerr<<"Please provide the model pcds"<<std::endl;
		return -1;
	}
	
	//generate some test hypotheses
	PostProcessing<pcl::PointXYZ>::HypothesesT test_hypotheses;
	
	Hypothesis<pcl::PointXYZ> h;
	h.cloud = *cloud;
	h.fitness = 0.0;
	h.pose = Eigen::Matrix4f::Identity();
	h.type = type;
	
	Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f inv_trans = Eigen::Matrix4f::Identity();
	Eigen::Vector4f p;
	pcl::compute3DCentroid<pcl::PointXYZ>(*cloud, p);
	trans.block(0,3,3,1) = -p.block(0,0,3,1);
	inv_trans.block(0,3,3,1) = p.block(0,0,3,1);
	
	test_hypotheses.push_back(h);
	R = Eigen::AngleAxisf(0.01*M_PI, Eigen::Vector3f::UnitX());
	T=Eigen::Matrix4f::Identity();
	T.block(0,0,3,3) = R;
	T = (inv_trans*T*trans);
	test_hypotheses[0].pose = T;
	pcl::transformPointCloud<pcl::PointXYZ>(h.cloud, test_hypotheses[0].cloud, T);
	visualizer.addPointCloud<pcl::PointXYZ>(test_hypotheses[0].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(test_hypotheses[0].cloud.makeShared(), 0.0, 0.0, 255.0), "test");
	visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "test");
	visualizer.spin();
	visualizer.removePointCloud("test");
	
	
	test_hypotheses.push_back(h);
	R = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
	T=Eigen::Matrix4f::Identity();
	T.block(0,0,3,3) = R;
	T = (inv_trans*T*trans);
	test_hypotheses[1].pose = T;
	pcl::transformPointCloud<pcl::PointXYZ>(h.cloud, test_hypotheses[1].cloud, T);
	visualizer.addPointCloud<pcl::PointXYZ>(test_hypotheses[1].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(test_hypotheses[1].cloud.makeShared(), 0.0, 0.0, 255.0), "test");
	visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "test");
	visualizer.spin();
	visualizer.removePointCloud("test");
	
	test_hypotheses.push_back(h);
	R = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ());
	T=Eigen::Matrix4f::Identity();
	T.block(0,0,3,3) = R;
	T = (inv_trans*T*trans);
	test_hypotheses[2].pose = T;
	pcl::transformPointCloud<pcl::PointXYZ>(h.cloud, test_hypotheses[2].cloud, T);
	visualizer.addPointCloud<pcl::PointXYZ>(test_hypotheses[2].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(test_hypotheses[2].cloud.makeShared(), 0.0, 0.0, 255.0), "test");
	visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "test");
	visualizer.spin();
	visualizer.removePointCloud("test");
	
	PostProcessing<pcl::PointXYZ> pp;
	pp.setInputCluster(cloud);
	pp.setInputHypotheses(test_hypotheses);
	pp.setInputModels(models);
	pp.verify();
	
	PostProcessing<pcl::PointXYZ>::HypothesesT good_hypotheses;
	PostProcessing<pcl::PointXYZ>::HypothesesT bad_hypotheses;
	
	pp.getGoodHypotheses(good_hypotheses);
	pp.getBadHypotheses(bad_hypotheses);
	
	//show the result
	std::cout<<"Get total "<<good_hypotheses.size()<<" good hypotheses\n";
	for(unsigned int i = 0; i < good_hypotheses.size(); ++i)
	{
		std::cout<<"Good hypothesis["<<i<<"]:\n"<<"fitness="<<good_hypotheses[i].fitness<<std::endl<<good_hypotheses[i].pose<<std::endl;
		visualizer.addPointCloud<pcl::PointXYZ>(good_hypotheses[i].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(good_hypotheses[i].cloud.makeShared(), 0.0, 0.0, 255.0), "aligned");
		visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "aligned");
		visualizer.spin();
		visualizer.removePointCloud("aligned");
	}
	
	std::cout<<"Get total "<<bad_hypotheses.size()<<" bad hypotheses\n";
	for(unsigned int i = 0; i < bad_hypotheses.size(); ++i)
	{
		std::cout<<"Bad hypothesis["<<i<<"]:\n"<<"fitness="<<bad_hypotheses[i].fitness<<std::endl<<bad_hypotheses[i].pose<<std::endl;
		visualizer.addPointCloud<pcl::PointXYZ>(bad_hypotheses[i].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(bad_hypotheses[i].cloud.makeShared(), 0.0, 0.0, 255.0), "aligned");
		visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "aligned");
		visualizer.spin();
		visualizer.removePointCloud("aligned");
	}
	
	return 0;
}
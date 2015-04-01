#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "pose_estimation/crh_pose_estimation.h"

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	//load test cloud
	std::string test;
	if(pcl::console::parse_argument(argc, argv, "-test", test) == -1)
	{
		std::cerr<<"Please specify the test cloud"<<std::endl;
		return -1;
	}
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>(test, *cloud);
	
	//generate some hypotheses
	Hypothesis<pcl::PointXYZ> hypothesis;
	
	hypothesis.fitness = 1.0;
	hypothesis.type = "test1";
	Eigen::Matrix3f R;
	R = Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(0.1 * M_PI, Eigen::Vector3f::UnitX());
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform.block(0,0,3,3) = R;
	
	pcl::transformPointCloud<pcl::PointXYZ>(*cloud, hypothesis.cloud, transform);
	hypothesis.pose = transform;
	
	pcl::visualization::PCLVisualizer visualizer("Result");
	visualizer.setBackgroundColor (0.0, 0.0, 0.0);
	visualizer.addCoordinateSystem (0.5);
	visualizer.registerKeyboardCallback(keyboardEventOccurred, (void*)&visualizer);
	visualizer.addPointCloud<pcl::PointXYZ>(cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud, 0.0, 255.0, 0.0), "scene");
	visualizer.addPointCloud<pcl::PointXYZ>(hypothesis.cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(hypothesis.cloud.makeShared(), 0.0, 0.0, 255.0), "model");
	visualizer.spin();
	visualizer.removeAllPointClouds();
	
	CRHPoseEstimation<pcl::PointXYZ> crh;
	crh.setInputScene(cloud);
	crh.setInputHypothesis(hypothesis);
	crh.estimate();
	
	std::vector<Hypothesis<pcl::PointXYZ> > hypotheses;
	crh.getResultHypotheses(hypotheses);
	
	for(unsigned int i = 0; i < hypotheses.size(); ++i)
	{
		std::cout<<hypotheses[i].pose<<std::endl;
		
		visualizer.addPointCloud<pcl::PointXYZ>(cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud, 0.0, 255.0, 0.0), "scene");
		visualizer.addPointCloud<pcl::PointXYZ>(hypotheses[i].cloud.makeShared(), pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(hypotheses[i].cloud.makeShared(), 0.0, 0.0, 255.0), "model");
		visualizer.spin();
		visualizer.removeAllPointClouds();
	}
	
	
	return 0;
}
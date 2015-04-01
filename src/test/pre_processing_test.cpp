#include "pre_processing/pre_processing.h"
#include <pcl/apps/3d_rec_framework/tools/openni_frame_source.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <iostream>

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	std::string roi;
	if(pcl::console::parse_argument(argc, argv, "-roi", roi) == -1)
	{
		std::cerr<<"Unable to read roi.txt"<<std::endl;
		return -1;
	}
	
	OpenNIFrameSource::OpenNIFrameSource camera("1@6");
	OpenNIFrameSource::PointCloudPtr new_frame;
	
	pcl::visualization::PCLVisualizer visualizer("Result");
	visualizer.setBackgroundColor (0.0, 0.0, 0.0);
	visualizer.addCoordinateSystem (0.5);
	visualizer.registerKeyboardCallback(keyboardEventOccurred, (void*)&visualizer);
	
	PreProcessing<pcl::PointXYZRGBA> pp;
	pp.setROIBoundary(roi);
	
	while(!visualizer.wasStopped())
	{
		pcl::ScopeTime t ("Frame rate:");
		
		visualizer.removeAllPointClouds();
		
		new_frame = camera.snap();
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr copied_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::copyPointCloud (*new_frame, *copied_cloud);
		
		visualizer.addPointCloud<pcl::PointXYZRGBA>(copied_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(copied_cloud), "scene");
		
		pp.setInputScene(copied_cloud);
		pp.process();
		pcl::PointCloud<pcl::PointXYZRGBA>::CloudVectorType clusters;
		pp.getResultClusters(clusters);
		
		for(unsigned int i = 0; i < clusters.size(); ++i)
		{
			std::stringstream cluster_name;
			cluster_name<<"cluster_"<<i;
			visualizer.addPointCloud<pcl::PointXYZRGBA>(clusters[i].makeShared(), pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZRGBA>(clusters[i].makeShared()), cluster_name.str());
			visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cluster_name.str());
		}
		visualizer.spinOnce();
	}
	
	return 0;
}
#include <pcl/console/parse.h>
#include <pcl/apps/3d_rec_framework/tools/openni_frame_source.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "global_recognition/global_recognition.h"

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	std::string db;
	if(pcl::console::parse_argument(argc, argv, "-db", db) == -1)
	{
		std::cerr<<"Can't not find -db option"<<std::endl;
		return -1;
	}
	
	std::string roi;
	if(pcl::console::parse_argument(argc, argv, "-roi", roi) == -1)
	{
		std::cerr<<"Can't not find -roi option"<<std::endl;
		return -1;
	}
	
	//OpenNIFrameSource::OpenNIFrameSource camera("1@6");
	OpenNIFrameSource::OpenNIFrameSource camera;
	OpenNIFrameSource::PointCloudPtr new_frame;
	sleep(1);
	
	pcl::visualization::PCLVisualizer visualizer("Result");
	visualizer.setBackgroundColor (0.0, 0.0, 0.0);
	visualizer.addCoordinateSystem (0.3);
	visualizer.registerKeyboardCallback(keyboardEventOccurred, (void*)&visualizer);
	
	GlobalRecognition<pcl::PointXYZ> gr;
	gr.init(db, roi);
	
	//while(!visualizer.wasStopped())
	while(true)
	{
		new_frame = camera.snap();
		pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud (*new_frame, *input_cloud);
		
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(input_cloud, 150.0, 150.0, 150.0);
		visualizer.addPointCloud<pcl::PointXYZ>(input_cloud, scene_color, "scene");
		visualizer.spinOnce();
		
		gr.setInputScene(input_cloud);
		gr.recognize();
		pcl::PointCloud<pcl::PointXYZ>::CloudVectorType clusters;
		gr.getClusters(clusters);
		GlobalRecognition<pcl::PointXYZ>::Hypotheses results;
		gr.getRecognitionResult(results);
		for(unsigned int i = 0; i < results.size(); ++i)
		{
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cluster_color(clusters[i].makeShared(), 0.0, 255.0, 0.0);
			visualizer.addPointCloud<pcl::PointXYZ>(clusters[i].makeShared(), cluster_color, "cluster");
			visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "cluster");
			visualizer.spinOnce();
			
			std::cout<<"Hypotheses for cluster["<<i<<"]:\n";
			//for(unsigned int j = 0; j < results[i].size(); ++j)
			//{
				std::cout<<"Type = "<<results[i][0].type<<std::endl;
				std::cout<<"Pose = \n"<<results[i][0].pose<<std::endl;
				
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> result_color(results[i][0].cloud.makeShared(), 0.0, 0.0, 255.0);
				visualizer.addPointCloud<pcl::PointXYZ>(results[i][0].cloud.makeShared(), result_color, "hypothesis");
				visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4.0, "hypothesis");
				visualizer.spin();
				visualizer.removePointCloud("hypothesis");
			//}
			
			visualizer.removePointCloud("cluster");
		}
		
		visualizer.removePointCloud("scene");
	}
		
	return 0;
}
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "recognition_database/recognition_database.h"

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	/*
	std::string model;
	if(pcl::console::parse_argument(argc, argv, "-model", model) == -1)
	{
		std::cerr<<"Please specify the model directory"<<std::endl;
		return -1;
	}
	*/
	std::string trained;
	if(pcl::console::parse_argument(argc, argv, "-trained", trained) == -1)
	{
		std::cerr<<"Please specify the trained directory"<<std::endl;
		return -1;
	}
	
	std::string query;
	if(pcl::console::parse_argument(argc, argv, "-query", query) == -1)
	{
		std::cerr<<"Please specify the database query"<<std::endl;
		return -1;
	}
	
	RecognitionDatabase<pcl::PointXYZ, pcl::ESFSignature640, flann::L1<float> > db;
	//db.trainDB(model, trained, true);
	db.setDescriptor("esf");
	db.loadDB(trained);
	
	pcl::PointCloud<pcl::ESFSignature640>::Ptr test_query(new pcl::PointCloud<pcl::ESFSignature640>);
	pcl::io::loadPCDFile<pcl::ESFSignature640>(query, *test_query);
	std::vector<std::vector<Hypothesis<pcl::PointXYZ> > > hypotheses;
	db.queryDB(test_query, 1, hypotheses);
	
	pcl::visualization::PCLVisualizer visualizer("Result");
	visualizer.setBackgroundColor (0.0, 0.0, 0.0);
	visualizer.addCoordinateSystem (0.5);
	visualizer.registerKeyboardCallback(keyboardEventOccurred, (void*)&visualizer);
	for(unsigned int i = 0; i < hypotheses[0].size(); ++i)
	{
		std::cout<<hypotheses[0][i].type<<std::endl;
		
		pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color_random(hypotheses[0][i].cloud.makeShared());
		visualizer.addPointCloud<pcl::PointXYZ>(hypotheses[0][i].cloud.makeShared(),color_random, "result");
		//visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "result");
		visualizer.spin();
		
		visualizer.removeAllPointClouds();
	}
	
	return 0;
}
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/search/impl/flann_search.hpp>

#include "recognition_database/recognition_database.h"

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){}

int main(int argc, char** argv)
{
	std::string model;
	if(pcl::console::parse_argument(argc, argv, "-model", model) == -1)
	{
		std::cerr<<"Please specify the model directory"<<std::endl;
		return -1;
	}
	
	std::string trained;
	if(pcl::console::parse_argument(argc, argv, "-train", trained) == -1)
	{
		std::cerr<<"Please specify the trained directory"<<std::endl;
		return -1;
	}
	
	RecognitionDatabase<pcl::PointXYZ, pcl::ESFSignature640, flann::L1<float> > db;
	db.trainDB(model, trained, true, 3);
	
	return 0;
}
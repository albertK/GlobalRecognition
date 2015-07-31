//copy paste from pcl_trunk
//pcl/apps/src/face_detection/openni_frame_source.cpp
//pcl/apps/include/pcl/apps/face_detection/openni_frame_source.h

#ifndef OPENNI_CAPTURE_H
#define OPENNI_CAPTURE_H

#include <boost/thread/mutex.hpp>
#include <boost/make_shared.hpp>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace OpenNIFrameSource
{
	//Declaration
	typedef pcl::PointXYZRGBA PointT;
	typedef pcl::PointCloud<PointT> PointCloud;
	typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
	typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
	
	/* A simple class for capturing data from an OpenNI camera */
	class PCL_EXPORTS OpenNIFrameSource
	{
	public:
		OpenNIFrameSource (const std::string& device_id = "");
		~OpenNIFrameSource ();
		
		const PointCloudPtr
		snap ();
		bool
		isActive ();
		void
		onKeyboardEvent (const pcl::visualization::KeyboardEvent & event);
		
	protected:
		void
		onNewFrame (const PointCloudConstPtr &cloud);
		
		pcl::OpenNIGrabber grabber_;
		PointCloudPtr most_recent_frame_;
		int frame_counter_;
		boost::mutex mutex_;
		bool active_;
	};
	
	//Definition
	OpenNIFrameSource::OpenNIFrameSource(const std::string& device_id) :
	grabber_ (device_id), most_recent_frame_ (), frame_counter_ (0), active_ (true)
	{
		boost::function<void(const PointCloudConstPtr&)> frame_cb = boost::bind (&OpenNIFrameSource::onNewFrame, this, _1);
		grabber_.registerCallback (frame_cb);
		grabber_.start ();
	}
	
	OpenNIFrameSource::~OpenNIFrameSource()
	{
		// Stop the grabber when shutting down
		grabber_.stop ();
	}
	
	bool OpenNIFrameSource::isActive()
	{
		return active_;
	}
	
	const PointCloudPtr OpenNIFrameSource::snap()
	{
		return (most_recent_frame_);
	}
	
	void OpenNIFrameSource::onNewFrame(const PointCloudConstPtr &cloud)
	{
		mutex_.lock ();
		++frame_counter_;
		most_recent_frame_ = boost::make_shared < PointCloud > (*cloud); // Make a copy of the frame
		mutex_.unlock ();
	}
	
	void OpenNIFrameSource::onKeyboardEvent(const pcl::visualization::KeyboardEvent & event)
	{
		// When the spacebar is pressed, trigger a frame capture
		mutex_.lock ();
		if (event.keyDown () && event.getKeySym () == "e")
		{
			active_ = false;
		}
		mutex_.unlock ();
	}
}

#endif
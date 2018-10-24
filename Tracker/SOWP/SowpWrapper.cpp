#include "SowpWrapper.h"
#include "Config.h"
#include "Tracker.h"

#include <opencv/highgui.h>



void SowpWrapper::Initialize(const cv::Mat& Frame, FloatRect& BBox)
{
	sowp::Config config;
	config.num_channel = 3;
	config.init_bbox.x = BBox.XMin();
	config.init_bbox.y = BBox.YMin();
	config.init_bbox.w = BBox.Width();
	config.init_bbox.h = BBox.Height();

	config.init_bbox.x -= 1; // Matlab -> C++
	config.init_bbox.y -= 1; // Matlab -> C++  

	if (config.init_bbox.w < config.init_bbox.h)
	{
		config.scale_width = config.init_bbox.w / 32.0;
		config.patch_width = std::round(config.init_bbox.w / (8.0*config.scale_width));
		config.patch_height = std::round(config.init_bbox.h / (8.0*config.scale_width));
		config.scale_height = config.init_bbox.h / (8.0*config.patch_height);
	}
	else
	{
		config.scale_height = config.init_bbox.h / 32.0;
		config.patch_height = std::round(config.init_bbox.h / (8.0*config.scale_height));
		config.patch_width = std::round(config.init_bbox.w / (8.0*config.scale_height));
		config.scale_width = config.init_bbox.w / (8.0*config.patch_width);
	}
	config.search_radius
		= sqrt(config.init_bbox.w*config.init_bbox.h / (config.scale_width*config.scale_height));

	if (config.num_channel == 3)
		config.image_type = cv::IMREAD_COLOR;
	else
		config.image_type = cv::IMREAD_GRAYSCALE;

	Core = new sowp::Tracker(config, Frame);
}

FloatRect SowpWrapper::Track(const cv::Mat& Frame)
{
	FloatRect bb=TrackWithoutUpdate(Frame);
	Update();
	return bb;
}

FloatRect SowpWrapper::TrackWithoutUpdate(const cv::Mat& frame)
{
	sowp::Rect bb = Core->track(frame, update);
	return FloatRect(bb.x, bb.y, bb.w, bb.h);
}

FloatRect SowpWrapper::TrackWithoutUpdate(const cv::Mat& frame, const FloatRect lastBb)
{
	return TrackWithoutUpdate(frame);
}

void SowpWrapper::Update()
{
	if (update)
	{
		Core->update();
	}
}

ITrackerWrapper* SowpWrapper::clone()
{
	SowpWrapper* mirror = new SowpWrapper;
	mirror->Core = new sowp::Tracker(*this->Core);
	return mirror;
}

SowpWrapper::SowpWrapper()
{
}


SowpWrapper::~SowpWrapper()
{
	delete Core;
}

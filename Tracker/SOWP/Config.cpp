#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>

#include "Config.h"

namespace sowp
{
	Config::Config()
	{

	};

	Config::Config(const Config &config)
		: /// Information
		sequence_name(config.sequence_name),
		sequence_path(config.sequence_path),
		result_path(config.result_path),        
		init_frame(config.init_frame),
		end_frame(config.end_frame),
		image_type(config.image_type),
		init_bbox(config.init_bbox),
		num_channel(config.num_channel),
		patch_width(config.patch_width),
		patch_height(config.patch_height),
		scale_width(config.scale_width),
		scale_height(config.scale_height),
		search_radius(config.search_radius)
	{
	}


	void Config::set(Config config)
	{
		/// Information
		sequence_name = config.sequence_name;
		sequence_path = config.sequence_path;
		result_path = config.result_path;  
		init_frame = config.init_frame;
		end_frame = config.end_frame;
		image_type = config.image_type;
		init_bbox.set(config.init_bbox);
		num_channel = config.num_channel;
		patch_width = config.patch_width;
		patch_height = config.patch_height;
		scale_width = config.scale_width;
		scale_height = config.scale_height;
		search_radius = config.search_radius;
	}
}

#pragma once
#include "Rect.h"

namespace sowp
{
	class Config
	{
	public:
		/// Information
		std::string sequence_name;
		std::string sequence_path; 
		std::string result_path;
		int init_frame;
		int end_frame;  
		int image_type;
		Rect init_bbox;
		int num_channel;  
		int patch_width;
		int patch_height;
		int search_radius;
		double scale_width;
		double scale_height;

		Config();
		Config(const Config &config);
		void set(Config config);
	};
}

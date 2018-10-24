#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>

#include "Config.h"
#include "Tracker.h"

int main(int argc, char* argv[])
{
  Config config;
  
  /// Generate configuration
  config.sequence_path = argv[1];
  config.sequence_name = argv[2];
  config.init_frame = std::stoi(argv[3]);
  config.end_frame = std::stoi(argv[4]);
  config.num_channel = std::stoi(argv[5]);  
  config.init_bbox.x = std::stod(argv[6]);
  config.init_bbox.y = std::stod(argv[7]);
  config.init_bbox.w = std::stod(argv[8]);
  config.init_bbox.h = std::stod(argv[9]);

  config.init_bbox.x -= 1; // Matlab -> C++
  config.init_bbox.y -= 1; // Matlab -> C++  

  config.result_path = std::string("./results/");

  if (config.init_bbox.w < config.init_bbox.h)
  {
    config.scale_width = config.init_bbox.w/32.0;
    config.patch_width = std::round(config.init_bbox.w/(8.0*config.scale_width));
    config.patch_height = std::round(config.init_bbox.h/(8.0*config.scale_width));  
    config.scale_height = config.init_bbox.h/(8.0*config.patch_height);
  }
  else
  {
    config.scale_height = config.init_bbox.h/32.0;      
    config.patch_height = std::round(config.init_bbox.h/(8.0*config.scale_height)); 
    config.patch_width = std::round(config.init_bbox.w/(8.0*config.scale_height));
    config.scale_width = config.init_bbox.w/(8.0*config.patch_width);
  }   
  config.search_radius 
    = sqrt(config.init_bbox.w*config.init_bbox.h/(config.scale_width*config.scale_height));

  if (config.num_channel == 3) 
    config.image_type = cv::IMREAD_COLOR; 
  else
    config.image_type = cv::IMREAD_GRAYSCALE;

  /// Tracking  
  srand(0);
  Tracker tracker(config);
  tracker.run();
  tracker.save();    

  return 0;
}

#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <string>
#include <queue>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "Rect.h"
#include "Config.h"
#include "StructuredSVM.h"

namespace sowp
{
	class Tracker
	{
	public:
		Tracker(const Tracker& t_Other)
			: NUM_PATCH(t_Other.NUM_PATCH),
			  CHANNEL_DIM(t_Other.CHANNEL_DIM),
			  ALPHA(t_Other.ALPHA),
			  GAMMA(t_Other.GAMMA),
			  THETA(t_Other.THETA),
			  RESTART_RATE(t_Other.RESTART_RATE),
			  NEIGHBOR_DIST_THRESHOLD(t_Other.NEIGHBOR_DIST_THRESHOLD),
			  IMAGE_TYPE(t_Other.IMAGE_TYPE),
			  INIT_FRAME(t_Other.INIT_FRAME),
			  END_FRAME(t_Other.END_FRAME),
			  NUM_FRAME(t_Other.NUM_FRAME),
			  NUM_CHANNEL(t_Other.NUM_CHANNEL),
			  PATCH_DIM(t_Other.PATCH_DIM),
			  OBJECT_DIM(t_Other.OBJECT_DIM),
			  SEQUENCE_NAME(t_Other.SEQUENCE_NAME),
			  SEQUENCE_PATH(t_Other.SEQUENCE_PATH),
			  RESULT_PATH(t_Other.RESULT_PATH),
			  INIT_BOX(t_Other.INIT_BOX),
			  patch_w(t_Other.patch_w),
			  patch_h(t_Other.patch_h),
			  scale_w(t_Other.scale_w),
			  scale_h(t_Other.scale_h),
			  search_r(t_Other.search_r),
			  patch_weight(t_Other.patch_weight),
			  prev_fore_prob(t_Other.prev_fore_prob),
			  prev_back_prob(t_Other.prev_back_prob),
			  image_bbox(t_Other.image_bbox),
			  border_bbox(t_Other.border_bbox),
			  feature_bbox(t_Other.feature_bbox),
			  object_bbox(t_Other.object_bbox),
			  classifier(t_Other.classifier),
			  feature_map(t_Other.feature_map),
			  result_box(t_Other.result_box),
			  current_result_box(t_Other.current_result_box),
			  patch_mask(t_Other.patch_mask),
			  expanded_patch_mask(t_Other.expanded_patch_mask),
			  image(t_Other.image),
			  image_channel(t_Other.image_channel),
			  integ_hist(t_Other.integ_hist)
		{
			for (cv::Mat t : t_Other.image_channel)
			{
				image_channel.push_back(t.clone());
			}
			for (cv::Mat t : t_Other.integ_hist)
			{
				integ_hist.push_back(t.clone());
			}
		}

	private:
		const int NUM_PATCH = 64;
		const int CHANNEL_DIM = 8;
		const double ALPHA = 35.0;
		const double GAMMA = 5.0;
		const double THETA = 0.3;
		const double RESTART_RATE = 0.75;
		const double NEIGHBOR_DIST_THRESHOLD = 1.10;

		const int IMAGE_TYPE;
		const int INIT_FRAME;
		const int END_FRAME;
		const int NUM_FRAME;
		const int NUM_CHANNEL;
		const int PATCH_DIM;
		const int OBJECT_DIM;
		const std::string SEQUENCE_NAME;
		const std::string SEQUENCE_PATH;
		const std::string RESULT_PATH;
		const Rect INIT_BOX;

	public:
		Tracker(Config& config);
		Tracker(Config& config, const cv::Mat& Frame);
		void run();
		void save();

		Rect track(const cv::Mat& Frame, bool& is_updated);
		void update();
		void initialize(const cv::Mat& Frame);
		void update_feature_map(const cv::Mat& Frame);


	private:
		void track(int t);
		void initialize();
		void initialize_bbox();
		void initialize_mask();
		void initialize_seed();

		bool update_object_box();
		void update_feature_map(int frame_id);
		void update_patch_weight();
		void update_classifier();
		void update_result_box(int frame_id);

		std::vector<double> compute_weight();
		void compute_color_histogram_map();
		void compute_gradient_histogram_map();
		void compute_feature_map();

		Eigen::VectorXd extract_test_feature(const Rect& sample);
		Eigen::VectorXd extract_train_feature(const Rect& sample);
		Eigen::VectorXd extract_patch_feature(const Rect& patch);
		Eigen::VectorXd extract_patch_feature(int x_min, int y_min, int x_max, int y_max);
		Eigen::VectorXd extract_expanded_patch_feature(const Rect& epatch);

		std::vector<Rect> extract_patch(Rect center);
		std::vector<Rect> extract_patch(Rect center, std::vector<Rect> mask);
		std::vector<Rect> extract_expanded_patch(Rect center);
		std::vector<Rect> extract_expanded_patch(Rect center, std::vector<Rect> expanded_mask);
		std::vector<Rect> extract_train_sample(Rect center);

		void display_result(int t, std::string window_name);

	private:
		int patch_w;
		int patch_h;
		double scale_w;
		double scale_h;
		int search_r;

		std::vector<double> patch_weight;
		Eigen::VectorXd prev_fore_prob;
		Eigen::VectorXd prev_back_prob;

		Rect image_bbox;
		Rect border_bbox;
		Rect feature_bbox;
		Rect object_bbox;

		StructuredSVM classifier;
		std::vector<Eigen::VectorXd> feature_map;
		std::vector<Rect> result_box;
		Rect current_result_box;

		std::vector<Rect> patch_mask;
		std::vector<Rect> expanded_patch_mask;

		cv::Mat image;
		std::vector<cv::Mat> image_channel;
		std::vector<cv::Mat> integ_hist;
	};
}
#endif TRACKER_H


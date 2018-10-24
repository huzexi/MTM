#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tracker.h"

namespace sowp
{
	Tracker::Tracker(Config& config) :
		/// Constant Setting
		SEQUENCE_NAME(config.sequence_name),
		SEQUENCE_PATH(config.sequence_path),
		RESULT_PATH(config.result_path),
		IMAGE_TYPE(config.image_type),
		INIT_FRAME(config.init_frame),
		END_FRAME(config.end_frame),
		NUM_FRAME(END_FRAME - INIT_FRAME + 1),
		NUM_CHANNEL(config.num_channel + 1),
		PATCH_DIM(NUM_CHANNEL * CHANNEL_DIM),
		OBJECT_DIM(NUM_PATCH * PATCH_DIM),
		INIT_BOX(config.init_bbox)
	{
		/// Variable Setting
		patch_w = config.patch_width;
		patch_h = config.patch_height;
		scale_w = config.scale_width;
		scale_h = config.scale_height;
		search_r = config.search_radius;

		char image_name[100];
		sprintf_s(image_name, 100, "%04d.jpg", INIT_FRAME);
		image = cv::imread(SEQUENCE_PATH + image_name, IMAGE_TYPE);
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		border_bbox.set(0, 0, image.cols - 1, image.rows - 1);
		image_bbox.set(patch_w, patch_h, border_bbox.w - 2 * patch_w, border_bbox.h - 2 * patch_h);

		object_bbox.x = std::round(config.init_bbox.x / scale_w) + patch_w;
		object_bbox.y = std::round(config.init_bbox.y / scale_h) + patch_h;
		object_bbox.w = std::round(config.init_bbox.w / scale_w);
		object_bbox.h = std::round(config.init_bbox.h / scale_h);

		if (object_bbox.x < 0)
			object_bbox.x = 0;
		if (object_bbox.x + object_bbox.w > image_bbox.x + image_bbox.w)
			object_bbox.x = image_bbox.x + image_bbox.w - object_bbox.w;
		if (object_bbox.y < 0)
			object_bbox.y = 0;
		if (object_bbox.y + object_bbox.h > image_bbox.y + image_bbox.h)
			object_bbox.y = image_bbox.y + image_bbox.h - object_bbox.h;

		result_box.resize(NUM_FRAME, Rect());
	}

	Tracker::Tracker(Config& config, const cv::Mat& Frame) :
		/// Constant Setting
		SEQUENCE_NAME(config.sequence_name),
		SEQUENCE_PATH(config.sequence_path),
		RESULT_PATH(config.result_path),
		IMAGE_TYPE(config.image_type),
		INIT_FRAME(config.init_frame),
		END_FRAME(config.end_frame),
		NUM_FRAME(END_FRAME - INIT_FRAME + 1),
		NUM_CHANNEL(config.num_channel + 1),
		PATCH_DIM(NUM_CHANNEL * CHANNEL_DIM),
		OBJECT_DIM(NUM_PATCH * PATCH_DIM),
		INIT_BOX(config.init_bbox)
	{
		/// Variable Setting
		patch_w = config.patch_width;
		patch_h = config.patch_height;
		scale_w = config.scale_width;
		scale_h = config.scale_height;
		search_r = config.search_radius;

		image = Frame;
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		border_bbox.set(0, 0, image.cols - 1, image.rows - 1);
		image_bbox.set(patch_w, patch_h, border_bbox.w - 2 * patch_w, border_bbox.h - 2 * patch_h);

		object_bbox.x = std::round(config.init_bbox.x / scale_w) + patch_w;
		object_bbox.y = std::round(config.init_bbox.y / scale_h) + patch_h;
		object_bbox.w = std::round(config.init_bbox.w / scale_w);
		object_bbox.h = std::round(config.init_bbox.h / scale_h);

		if (object_bbox.x < 0)
			object_bbox.x = 0;
		if (object_bbox.x + object_bbox.w > image_bbox.x + image_bbox.w)
			object_bbox.x = image_bbox.x + image_bbox.w - object_bbox.w;
		if (object_bbox.y < 0)
			object_bbox.y = 0;
		if (object_bbox.y + object_bbox.h > image_bbox.y + image_bbox.h)
			object_bbox.y = image_bbox.y + image_bbox.h - object_bbox.h;

//		result_box.resize(NUM_FRAME,Rect());
		current_result_box = object_bbox;

		initialize(Frame);

	}

	void Tracker::display_result(int t, std::string window_name)
	{
		int frame_id = t + INIT_FRAME;

		char image_name[100];
		sprintf_s(image_name, 100, "%04d.jpg", frame_id);
		image = cv::imread(SEQUENCE_PATH + image_name, cv::IMREAD_COLOR);
		cv::rectangle(image,
		              cv::Rect((int)result_box[t].x, (int)result_box[t].y, (int)result_box[t].w, (int)result_box[t].h),
		              CV_RGB(255, 255, 0),
		              2);

		cv::imshow(window_name, image);
		cv::waitKey(1);
	}

	void Tracker::run()
	{
		std::cout << "[Sequence] " << SEQUENCE_NAME << '\n';
		std::cout << "-------------------------------------------- \n";

		/// Initialize 
		std::cout << "Start initialization\n";
		initialize();
		display_result(0, "result");
		std::cout << "Complete initialization \n";
		std::cout << "-------------------------------------------- \n";

		/// track
		std::cout << "Start tracking\n";
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int t = 1; t < NUM_FRAME; ++t)
		{
			int frame_id = t + INIT_FRAME;
			track(frame_id);
			display_result(t, "result");
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		auto fps = (double)(NUM_FRAME - 1.0) / std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
		std::cout << "FPS: " << fps << "\n";

		std::cout << "Complete tracking \n";
		std::cout << "-------------------------------------------- \n\n";
	}

	void Tracker::save()
	{
		std::ofstream ofs(RESULT_PATH + SEQUENCE_NAME + "_swop.txt");

		for (int i = 0; i < result_box.size(); ++i)
		{
			char temp[100];
			sprintf_s(temp, 100, "%.2lf %.2lf %.2lf %.2f",
			          result_box[i].x + 1, result_box[i].y + 1, result_box[i].w, result_box[i].h);
			ofs << temp << '\n';
		}
		ofs.close();
	}

	Rect Tracker::track(const cv::Mat& Frame, bool& is_updated)
	{
		image = Frame;
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		update_feature_map(Frame);
		is_updated = update_object_box();

//		update_result_box(frame_id);
//		int t = frame_id - INIT_FRAME;
		current_result_box.x = (object_bbox.x - patch_w) * scale_w;
		current_result_box.y = (object_bbox.y - patch_h) * scale_h;
		current_result_box.w = object_bbox.w * scale_w;
		current_result_box.h = object_bbox.h * scale_h;

		return current_result_box;
	}

	void Tracker::update()
	{
		update_patch_weight();
		update_classifier();
	}

	void Tracker::initialize(const cv::Mat& Frame)
	{
		update_feature_map(Frame);
		initialize_mask();
		initialize_seed();
		update_patch_weight();
		update_classifier();
//		update_result_box(INIT_FRAME);
	}

	void Tracker::update_feature_map(const cv::Mat& Frame)
	{
		image = Frame;
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		image_channel.clear();
		int num_color_channel = NUM_CHANNEL - 1;
		for (int i = 0; i < num_color_channel; ++i)
		{ 
			image_channel.push_back(cv::Mat(image.rows, image.cols, CV_8UC1));
		}

		integ_hist.clear();
		int color_dim = PATCH_DIM - CHANNEL_DIM;
		for (int i = 0; i < color_dim; ++i)
		{ 
			integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_32SC1));
		}
		for (int i = 0; i < CHANNEL_DIM; ++i)
		{ 
			integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_64FC1));
		}

		feature_map.resize(image.rows * image.cols, Eigen::VectorXd::Zero(PATCH_DIM));
		cv::split(image, image_channel);

		compute_color_histogram_map();
		compute_gradient_histogram_map();
		compute_feature_map();
	}

	void Tracker::initialize()
	{
		std::cout << "- initialize feature map\n";
		update_feature_map(INIT_FRAME);

		std::cout << "- initialize patch weight\n";
		initialize_mask();
		initialize_seed();
		update_patch_weight();

		std::cout << "- initialize classifier\n";
		update_classifier();

		update_result_box(INIT_FRAME);
	}

	void Tracker::track(int frame_id)
	{
		update_feature_map(frame_id);
		bool is_updated = update_object_box();
		if (is_updated)
		{
			update_patch_weight();
			update_classifier();
		}

		update_result_box(frame_id);
	}

	void Tracker::update_result_box(int frame_id)
	{
		int t = frame_id - INIT_FRAME;
		result_box[t].x = (object_bbox.x - patch_w) * scale_w;
		result_box[t].y = (object_bbox.y - patch_h) * scale_h;
		result_box[t].w = object_bbox.w * scale_w;
		result_box[t].h = object_bbox.h * scale_h;
	}

	void Tracker::initialize_mask()
	{
		patch_mask.clear();
		std::vector<Rect> patch = extract_patch(object_bbox);
		for (int i = 0; i < patch.size(); ++i)
			patch_mask.push_back(Rect(patch[i].x - object_bbox.x,
			                          patch[i].y - object_bbox.y,
			                          patch[i].w,
			                          patch[i].h));

		expanded_patch_mask.clear();
		std::vector<Rect> expanded_patch = extract_expanded_patch(object_bbox);
		for (int i = 0; i < expanded_patch.size(); ++i)
			expanded_patch_mask.push_back(Rect(expanded_patch[i].x - object_bbox.x,
			                                   expanded_patch[i].y - object_bbox.y,
			                                   expanded_patch[i].w,
			                                   expanded_patch[i].h));
	}

	void Tracker::initialize_seed()
	{
		std::vector<Rect> expanded_patch = extract_expanded_patch(object_bbox, expanded_patch_mask);
		Rect s_bbox(object_bbox.x + 0.1f * object_bbox.w,
		            object_bbox.y + 0.1f * object_bbox.h,
		            0.8f * object_bbox.w,
		            0.8f * object_bbox.h);

		prev_fore_prob = Eigen::VectorXd::Zero(expanded_patch.size());
		prev_back_prob = Eigen::VectorXd::Zero(expanded_patch.size());
		for (int i = 0; i < expanded_patch.size(); ++i)
		{
			if (expanded_patch[i].is_inside(s_bbox))
				prev_fore_prob(i) = 1;
			if (!expanded_patch[i].is_inside(object_bbox))
				prev_back_prob(i) = 1;
		}

		patch_weight.resize(NUM_PATCH, 0.0);
	}

	bool Tracker::update_object_box()
	{
		bool is_updated = false;

		Rect sample(object_bbox);
		Rect best_sample(object_bbox);
		double best_score = -DBL_MAX;

		for (int iy = -search_r; iy <= search_r; ++iy)
		{
			for (int ix = -search_r; ix <= search_r; ++ix)
			{
				sample.x = (int)object_bbox.x + ix;
				sample.y = (int)object_bbox.y + iy;

				if (!sample.is_inside(image_bbox))
					continue;

				Eigen::VectorXd sample_feature = extract_test_feature(sample);
				double score = classifier.test(sample_feature);
				if (score > best_score)
				{
					best_score = score;
					best_sample.set(sample);
				}
			}
		}

		Eigen::VectorXd best_sample_feature = extract_test_feature(best_sample);
		double validation_score = classifier.validation_test(best_sample_feature);
		if (validation_score > THETA)
		{
			is_updated = true;
			object_bbox.set(best_sample);
		}

		return is_updated;
	}

	void Tracker::update_feature_map(int idx)
	{
		char image_name[100];
		sprintf_s(image_name, 100, "%04d.jpg", idx);
		image = cv::imread(SEQUENCE_PATH + image_name, IMAGE_TYPE);
		cv::resize(image, image, cv::Size(), 1 / scale_w, 1 / scale_h);
		cv::copyMakeBorder(image, image, patch_h, patch_h, patch_w, patch_w, cv::BORDER_CONSTANT, cv::Scalar());

		image_channel.clear();
		int num_color_channel = NUM_CHANNEL - 1;
		for (int i = 0; i < num_color_channel; ++i)
			image_channel.push_back(cv::Mat(image.rows, image.cols, CV_8UC1));

		integ_hist.clear();
		int color_dim = PATCH_DIM - CHANNEL_DIM;
		for (int i = 0; i < color_dim; ++i)
			integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_32SC1));
		for (int i = 0; i < CHANNEL_DIM; ++i)
			integ_hist.push_back(cv::Mat(image.rows + 1, image.cols + 1, CV_64FC1));

		feature_map.resize(image.rows * image.cols, Eigen::VectorXd::Zero(PATCH_DIM));
		cv::split(image, image_channel);

		compute_color_histogram_map();
		compute_gradient_histogram_map();
		compute_feature_map();
	}

	void Tracker::update_patch_weight()
	{
		std::vector<Rect> expanded_patch = extract_expanded_patch(object_bbox, expanded_patch_mask);
		std::vector<Eigen::VectorXd> expanded_feature(expanded_patch.size(), Eigen::VectorXd(OBJECT_DIM));
		for (int i = 0; i < expanded_patch.size(); ++i)
			expanded_feature[i] = extract_expanded_patch_feature(expanded_patch[i]);

		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(expanded_patch.size(), expanded_patch.size());
		Eigen::MatrixXd r = Eigen::MatrixXd::Zero(expanded_patch.size(), 2);
		Eigen::MatrixXd p_new = Eigen::MatrixXd::Zero(expanded_patch.size(), 2);
		Eigen::MatrixXd p_old = Eigen::MatrixXd::Zero(expanded_patch.size(), 2);

		Rect s_bbox(object_bbox.x + 0.1f * object_bbox.w,
		            object_bbox.y + 0.1f * object_bbox.h,
		            0.8f * object_bbox.w,
		            0.8f * object_bbox.h);

		int oidx = -1;
		for (int i = 0; i < expanded_patch.size(); ++i)
		{
			for (int j = i; j < expanded_patch.size(); ++j)
			{
				double dist1 = abs(expanded_patch[i].x - expanded_patch[j].x) / expanded_patch[i].w;
				double dist2 = abs(expanded_patch[i].y - expanded_patch[j].y) / expanded_patch[i].h;

				if (dist1 > NEIGHBOR_DIST_THRESHOLD || dist2 > NEIGHBOR_DIST_THRESHOLD)
					continue;

				double similarity = exp(-GAMMA * ((expanded_feature[i] - expanded_feature[j]).squaredNorm()));

				A(i, j) = similarity;
				A(j, i) = similarity;
			}

			if (expanded_patch[i].is_inside(s_bbox))
			{
				r(i, 0) = prev_fore_prob(i);
			}
			else if (expanded_patch[i].is_inside(object_bbox))
			{
				r(i, 0) = prev_fore_prob(i);
				r(i, 1) = prev_back_prob(i);
			}
			else
			{
				r(i, 1) = prev_back_prob(i);
			}
		}

		for (int i = 0; i < expanded_patch.size(); ++i)
		{
			A.col(i) = A.col(i) / A.col(i).sum();
			p_old(i, 0) = 0.01 * (double)(std::rand() % 100);
			p_old(i, 1) = 0.01 * (double)(std::rand() % 100);
		}

		p_old.col(0) = p_old.col(0) / p_old.col(0).sum();
		p_old.col(1) = p_old.col(1) / p_old.col(1).sum();
		r.col(0) = r.col(0) / r.col(0).sum();
		r.col(1) = r.col(1) / r.col(1).sum();

		while (1)
		{
			p_new = (1 - RESTART_RATE) * A * p_old + RESTART_RATE * r;
			p_new.col(0) = p_new.col(0) / p_new.col(0).sum();
			p_new.col(1) = p_new.col(1) / p_new.col(1).sum();

			if ((p_new - p_old).norm() < 1e-6)
				break;

			p_old = p_new;
		}

		prev_fore_prob = p_new.col(0);
		prev_back_prob = p_new.col(1);

		int idx = 0;
		for (int i = 0; i < expanded_patch.size(); ++i)
		{
			if (expanded_patch[i].is_inside(object_bbox))
			{
				patch_weight[idx] = 1 / (1 + exp(-ALPHA * (prev_fore_prob(i) - prev_back_prob(i))));
				++idx;
			}
		}
	}

	void Tracker::update_classifier()
	{
		std::vector<Rect> train_sample = extract_train_sample(object_bbox);
		std::vector<Eigen::VectorXd> train_features(train_sample.size(), Eigen::VectorXd(OBJECT_DIM));
		for (int i = 0; i < train_sample.size(); ++i)
			train_features[i] = extract_train_feature(train_sample[i]);

		classifier.train(train_sample, train_features, 0);
	}

	Eigen::VectorXd Tracker::extract_test_feature(const Rect& sample)
	{
		Eigen::VectorXd feature(Eigen::VectorXd::Zero(OBJECT_DIM));
		for (int i = 0; i < NUM_PATCH; ++i)
		{
			int x_min = sample.x + patch_mask[i].x;
			int y_min = sample.y + patch_mask[i].y;

			feature.segment(i * PATCH_DIM, PATCH_DIM) = patch_weight[i] * feature_map[image.cols * y_min + x_min];
		}
		feature.normalize();
		return feature;
	}

	Eigen::VectorXd Tracker::extract_train_feature(const Rect& sample)
	{
		Eigen::VectorXd feature(Eigen::VectorXd::Zero(OBJECT_DIM));
		for (int j = 0; j < NUM_PATCH; ++j)
		{
			int x_min = sample.x + patch_mask[j].x;
			int y_min = sample.y + patch_mask[j].y;

			Rect r(x_min, y_min, patch_w, patch_h);
			if (r.is_inside(feature_bbox))
				feature.segment(j * PATCH_DIM, PATCH_DIM) = patch_weight[j] * feature_map[image.cols * y_min + x_min];
			else
				feature.segment(j * PATCH_DIM, PATCH_DIM) = patch_weight[j] * extract_patch_feature(r);
		}
		feature.normalize();
		return feature;
	}

	Eigen::VectorXd Tracker::extract_patch_feature(const Rect& patch)
	{
		int x_min = (int)patch.x;
		int y_min = (int)patch.y;
		int x_max = (int)(patch.x + patch.w);
		int y_max = (int)(patch.y + patch.h);

		return extract_patch_feature(x_min, y_min, x_max, y_max);
	}

	Eigen::VectorXd Tracker::extract_patch_feature(int x_min, int y_min, int x_max, int y_max)
	{
		Eigen::VectorXd feature(PATCH_DIM);

		int color_dim = PATCH_DIM - CHANNEL_DIM;
		double patch_area = patch_w * patch_h;
		for (int i = 0; i < color_dim; ++i)
		{
			double sum = integ_hist[i].at<int>(y_min, x_min)
				+ integ_hist[i].at<int>(y_max, x_max)
				- integ_hist[i].at<int>(y_max, x_min)
				- integ_hist[i].at<int>(y_min, x_max);
			feature[i] = sum / patch_area;
		}

		double total_sum = 0;
		int grad_dim = CHANNEL_DIM;
		for (int i = 0; i < grad_dim; ++i)
		{
			double sum = integ_hist[color_dim + i].at<double>(y_min, x_min)
				+ integ_hist[color_dim + i].at<double>(y_max, x_max)
				- integ_hist[color_dim + i].at<double>(y_max, x_min)
				- integ_hist[color_dim + i].at<double>(y_min, x_max);

			feature[color_dim + i] = sum;
			total_sum += sum;
		}
		for (int i = 0; i < grad_dim; ++i)
			feature[color_dim + i] /= (total_sum + 1e-6);

		return feature;
	}

	Eigen::VectorXd Tracker::extract_expanded_patch_feature(const Rect& patch)
	{
		Eigen::VectorXd feature;
		if (patch.is_inside(feature_bbox))
			feature = feature_map[image.cols * patch.y + patch.x];
		else
			feature = extract_patch_feature(patch);

		return feature;
	}

	void Tracker::compute_color_histogram_map()
	{
		double bin_size = 32.0;
		int num_color_channel = NUM_CHANNEL - 1;

		for (int i = 0; i < num_color_channel; ++i)
		{
			cv::Mat tmp(image.rows, image.cols, CV_8UC1);
			tmp.setTo(0);

			for (int j = 0; j < CHANNEL_DIM; ++j)
			{
				for (int y = 0; y < image.rows; ++y)
				{
					const uchar* src = image_channel[i].ptr(y);
					uchar* dst = tmp.ptr(y);
					for (int x = 0; x < image.cols; ++x)
					{
						int bin = (int)((double)(*src) / bin_size);
						*dst = (bin == j) ? 1 : 0;
						++src;
						++dst;
					}
				}

				cv::integral(tmp, integ_hist[i * CHANNEL_DIM + j]);
			}
		}
	}

	void Tracker::compute_gradient_histogram_map()
	{
		float bin_size = 22.5;
		float radian_to_degree = 180.0 / CV_PI;

		cv::Mat gray_image(image.rows, image.cols, CV_8UC1);
		if (IMAGE_TYPE == cv::IMREAD_COLOR)
			cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		else
			image.copyTo(gray_image);

		cv::Mat x_sobel, y_sobel;
		cv::Sobel(gray_image, x_sobel, CV_32FC1, 1, 0);
		cv::Sobel(gray_image, y_sobel, CV_32FC1, 0, 1);

		std::vector<cv::Mat> bins;
		for (int i = 0; i < CHANNEL_DIM; ++i)
			bins.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32FC1));

		for (int y = 0; y < image.rows; ++y)
		{
			float* x_sobel_row_ptr = (float*)(x_sobel.row(y).data);
			float* y_sobel_row_ptr = (float*)(y_sobel.row(y).data);

			std::vector<float*> bins_row_ptrs(CHANNEL_DIM, nullptr);
			for (int i = 0; i < CHANNEL_DIM; ++i)
				bins_row_ptrs[i] = (float*)(bins[i].row(y).data);

			for (int x = 0; x < image.cols; ++x)
			{
				if (x_sobel_row_ptr[x] == 0)
					x_sobel_row_ptr[x] += 0.00001;

				float orientation = atan(y_sobel_row_ptr[x] / x_sobel_row_ptr[x]) * radian_to_degree + 90;
				float magnitude = sqrt(x_sobel_row_ptr[x] * x_sobel_row_ptr[x] + y_sobel_row_ptr[x] * y_sobel_row_ptr[x]);

				for (int i = 1; i < CHANNEL_DIM; ++i)
				{
					if (orientation <= bin_size * i)
					{
						bins_row_ptrs[i - 1][x] = magnitude;
						break;
					}
				}
			}
		}

		int color_dim = PATCH_DIM - CHANNEL_DIM;
		for (int i = 0; i < CHANNEL_DIM; ++i)
			cv::integral(bins[i], integ_hist[color_dim + i]);
	}

	void Tracker::compute_feature_map()
	{
		int x_start = -search_r;
		int x_end = search_r + object_bbox.w;
		int y_start = -search_r;
		int y_end = search_r + object_bbox.h;

		feature_bbox.set(object_bbox.x + x_start, object_bbox.y + y_start, x_end - x_start, y_end - y_start);

		for (int iy = y_start; iy <= y_end; ++iy)
		{
			int y_min = (int)object_bbox.y + iy;
			int y_max = y_min + patch_h;

			if ((y_min < border_bbox.y) || (y_max > border_bbox.y + border_bbox.h))
			{
				continue;
			}

			for (int ix = x_start; ix <= x_end; ++ix)
			{
				int x_min = (int)object_bbox.x + ix;
				int x_max = x_min + patch_w;
				if ((x_min < border_bbox.x) || (x_max > border_bbox.x + border_bbox.w))
				{
					continue;
				}

				feature_map[image.cols * y_min + x_min] = extract_patch_feature(x_min, y_min, x_max, y_max);
			}
		}
	}

	std::vector<Rect> Tracker::extract_patch(Rect sample)
	{
		std::vector<Rect> patch;

		int x_counter = (int)(sample.w / (double)patch_w);
		int y_counter = (int)(sample.h / (double)patch_h);

		int dx = (int)(0.5 * (sample.w - x_counter * patch_w));
		int dy = (int)(0.5 * (sample.h - y_counter * patch_h));

		int x_start = (int)sample.x;
		int y_start = (int)sample.y;

		for (int iy = 0; iy < y_counter; ++iy)
		{
			int y = y_start + iy * (patch_h);
			for (int ix = 0; ix < x_counter; ++ix)
			{
				int x = x_start + ix * (patch_w);
				patch.push_back(Rect(x, y, patch_w, patch_h));
			}
		}

		return patch;
	}

	std::vector<Rect> Tracker::extract_patch(Rect center, std::vector<Rect> mask)
	{
		std::vector<Rect> samples;
		for (int i = 0; i < mask.size(); ++i)
		{
			int px_min = (int)(center.x + mask[i].x);
			int py_min = (int)(center.y + mask[i].y);
			int px_max = (int)(px_min + mask[i].w);
			int py_max = (int)(py_min + mask[i].h);

			Rect sample((float)px_min, (float)py_min, mask[i].w, mask[i].h);
			samples.push_back(sample);
		}

		return samples;
	}

	std::vector<Rect> Tracker::extract_expanded_patch(Rect sample)
	{
		std::vector<Rect> expanded_patch;

		int x_counter = (int)(sample.w / (double)patch_w);
		int y_counter = (int)(sample.h / (double)patch_h);

		int x_start = (int)sample.x;
		int y_start = (int)sample.y;

		for (int iy = -1; iy < y_counter + 1; ++iy)
		{
			int y = y_start + iy * (patch_h);
			for (int ix = -1; ix < x_counter + 1; ++ix)
			{
				int x = x_start + ix * (patch_w);
				expanded_patch.push_back(Rect(x, y, patch_w, patch_h));
			}
		}

		return expanded_patch;
	}

	std::vector<Rect> Tracker::extract_expanded_patch(Rect center, std::vector<Rect> expanded_mask)
	{
		std::vector<Rect> expanded_patch;
		for (int i = 0; i < expanded_mask.size(); ++i)
		{
			int px_min = (int)(center.x + expanded_mask[i].x);
			int py_min = (int)(center.y + expanded_mask[i].y);

			Rect patch(px_min, py_min, expanded_mask[i].w, expanded_mask[i].h);

			if (patch.is_inside(border_bbox))
				expanded_patch.push_back(patch);
		}

		return expanded_patch;
	}

	std::vector<Rect> Tracker::extract_train_sample(Rect center)
	{
		int num_r = 5;
		int num_t = 16;
		double radius = 2 * search_r;
		double rstep = radius / 5.0;
		double tstep = 2.0 * CV_PI / 16.0;

		std::vector<Rect> train_sample;
		train_sample.push_back(center);
		for (int ir = 1; ir <= num_r; ++ir)
		{
			double phase = (ir % 2) * tstep / 2;
			for (int it = 0; it < num_t; ++it)
			{
				double dx = ir * rstep * cosf(it * tstep + phase);
				double dy = ir * rstep * sinf(it * tstep + phase);

				Rect sample(center);
				sample.x = center.x + dx;
				sample.y = center.y + dy;

				if (sample.is_inside(image_bbox))
					train_sample.push_back(sample);
			}
		}

		return train_sample;
	}
}


#pragma once
#include "../ITrackerWrapper.h"
#include "kcftracker.hpp"

class KcfWrapper :
	public ITrackerWrapper
{
public:
	KcfWrapper();
	~KcfWrapper() override;

	void ConfigFeature(bool hog, bool lab, bool his);

	void Init(const cv::Mat &Frame, const IntRect &BBox) override;
	IntRect Track(const cv::Mat& Frame) override;

	IntRect TrackWithoutUpdate(const cv::Mat& frame) override;
	void Update() override;
	ITrackerWrapper* Clone() override;

private:
	static cv::Rect ResultPostProc(cv::Rect res, const cv::Mat& frm);

	kcf::KCFTracker *Core=nullptr;
	cv::Mat Frame;

	bool HOG = true;
	bool LAB = false;
	bool HIS = false;
};


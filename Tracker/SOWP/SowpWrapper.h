#pragma once
#include "../ITrackerWrapper.h"
#include "Tracker.h"

class SowpWrapper :
	public ITrackerWrapper
{
public:
	void Initialize(const cv::Mat& Frame, FloatRect& BBox) override;
	FloatRect Track(const cv::Mat& Frame) override;

	FloatRect TrackWithoutUpdate(const cv::Mat& frame) override;
	FloatRect TrackWithoutUpdate(const cv::Mat& frame, const FloatRect lastBb) override;
	void Update() override;
	ITrackerWrapper* clone() override;
	SowpWrapper();
	~SowpWrapper();
private:
	sowp::Tracker *Core;

	bool update;
};


#pragma once
#include "../ITrackerWrapper.h"
#include "Tracker.h"


class StruckWrapper :
	public ITrackerWrapper
{
public:
	StruckWrapper();
	~StruckWrapper();

	void Init(const cv::Mat &frame, const IntRect &bbox) override;
	IntRect Track(const cv::Mat& frame) override;
	IntRect TrackWithoutUpdate(const cv::Mat& frame) override;
	void Update() override;

	ITrackerWrapper* Clone() override;

private:
    Tracker *Core= nullptr;
    bool NeedUpdate=false;
    ImageRep* UpdateImage= nullptr;
};


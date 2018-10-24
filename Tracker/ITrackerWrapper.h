// Single Tracker
#pragma once
#include "../Rect.h"
#include <opencv/cv.h>

class ITrackerWrapper
{
public:
	std::string Name;

	ITrackerWrapper() {};
	virtual ~ITrackerWrapper()	{};

	virtual void Init(const cv::Mat &Frame, const IntRect &BBox) = 0; /// Called in first frame
	virtual IntRect Track(const cv::Mat& Frame) = 0; /// Track and update
	virtual IntRect TrackWithoutUpdate(const cv::Mat& frame) = 0;
	virtual void Update() = 0;
	virtual ITrackerWrapper* Clone() = 0;
};

class TestTrackerWrapper : public ITrackerWrapper
{
public:
	

	IntRect Track(const cv::Mat& Frame) override
	{
		IntRect bb=TrackWithoutUpdate(Frame);
		Update();
		return bb;
	};

	IntRect TrackWithoutUpdate(const cv::Mat& frame) override
	{
		bb.SetXMin(bb.XMin() - 5 <= 0 ? rand() % 200 + 1 : bb.XMin() - 5);
		bb.SetYMin(bb.YMin() - 5 <= 0 ? rand() % 200 + 1 : bb.YMin() - 5);
		//		bb.SetWidth(bb.Width() - 1<=0 ? rand() % 100 + 1 : bb.Width() - 1);
		//		bb.SetHeight(bb.Height() - 1 <= 0 ? rand() % 100 + 1 : bb.Height() - 1);
		return bb;
	};

	void Init(const cv::Mat &Frame, const IntRect &BBox) override
	{
		bb = BBox;
	};
	void Update() override
	{

	};
	ITrackerWrapper* Clone() override
	{
		return new TestTrackerWrapper(*this);
	};
	IntRect bb;
};

#pragma once


#include "../../ITrackerWrapper.h"
#include "dsst_tracker.hpp"

class DsstWrapper : public ITrackerWrapper {
public:
    DsstWrapper();

    ~DsstWrapper() override;

    void Init(const cv::Mat &Frame, const IntRect &bbox) override;
    IntRect Track(const cv::Mat& frame) override;

    IntRect TrackWithoutUpdate(const cv::Mat& frame) override;
    void Update() override;
    ITrackerWrapper* Clone() override;

private:
    cf_tracking::DsstParameters Paras;
    cf_tracking::DsstTracker *Core=nullptr;
    static cv::Rect ResultPostProc(cv::Rect res, const cv::Mat& frm);
    cv::Rect Result;

};

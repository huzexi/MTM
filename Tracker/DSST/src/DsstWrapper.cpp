//
// Created by zeke on 19/9/18.
//

#include <DsstWrapper.h>

#include "DsstWrapper.h"

DsstWrapper::DsstWrapper() {}

DsstWrapper::~DsstWrapper() {
    if(Core!= nullptr) {
        delete Core;
    }
}

void DsstWrapper::Init(const cv::Mat &frame, const IntRect &bbox) {
    Core = new cf_tracking::DsstTracker(Paras);
    Result = cv::Rect(bbox.XMin(), bbox.YMin(), bbox.Width(), bbox.Height());
    Core->reinit(frame, Result);
}

IntRect DsstWrapper::Track(const cv::Mat &frame) {
    Core->update(frame, Result);
    return ResultPostProc(Result, frame);
}

IntRect DsstWrapper::TrackWithoutUpdate(const cv::Mat &frame) {
    Core->update(frame, Result, false);
    return ResultPostProc(Result, frame);
}

void DsstWrapper::Update() {
    Core->updateModel(Core->Image, Core->NewPos, Core->NewScale);
}

ITrackerWrapper *DsstWrapper::Clone() {
    auto* mirror = new DsstWrapper(*this);
    *mirror=*this;
    mirror->Core = new cf_tracking::DsstTracker(*this->Core);
    return mirror;
}

cv::Rect DsstWrapper::ResultPostProc(cv::Rect res, const cv::Mat &frm) {
    res.x = std::max(0, res.x);
    res.y = std::max(0, res.y);

    res.x = std::min(frm.cols - 8, res.x);
    res.y = std::min(frm.rows - 8, res.y);

    res.width = std::max(res.width, 8);
    res.height = std::max(res.height, 8);

    res.width = std::min(res.width, frm.cols);
    res.height = std::min(res.height, frm.rows);

    if(res.x + res.width > frm.cols) {
        res.x = frm.cols - res.width;
    }
    if(res.y + res.height > frm.rows) {
        res.y = frm.rows - res.height;
    }
    return res;
}

#include "TrajectoriesRefiner.h"
#include "ResultMonitor.h"
#include <opencv/highgui.h>

vector<RectVec> TrajectoriesRefiner::TrackBack(
	const vector<ITrackerWrapper*>& trkList, 
	const vector<RectVec>& trjList, 
	const vector<Mat>& frmList
) const
{
	assert(trkList.size() == trjList.size());
	assert(trjList.front().size() == frmList.size());

	size_t interval = trjList.front().size();
	size_t trkN = trkList.size();
	vector<RectVec> trjBkList(trkN, RectVec(interval));

	// For displaying
	ResultMonitor monitor;
	monitor.SetWindowName("Track Back");

	// Copy trackers to trace back
	vector<ITrackerWrapper*> mirrorList;
	vector<ITrackerWrapper*> seedTrks=GetNewTrackers();

	for (int trkId = 0; trkId < trkN; trkId++)
	{
		ITrackerWrapper* bkTrk = seedTrks[trkId%seedTrks.size()];
//		ITrackerWrapper* bkTrk = GetNewTracker();

		IntRect bb = trjList[trkId].back();
		const Mat& frame = frmList.back();
		if (bb.Width() < 4)
		{
			bb.SetWidth(4);
		}
		if (bb.Height() < 4)
		{
			bb.SetHeight(4);
		}
		if (bb.Width()>frame.cols)
		{
			bb.SetXMin(0);
			bb.SetWidth(frame.cols);
		}
		if (bb.Height()>frame.rows)
		{
			bb.SetYMin(0);
			bb.SetHeight(frame.rows);
		}
		if (bb.XMax()>=frame.cols)
		{
			bb.SetXMin(frame.cols - bb.Width());
		}
		if (bb.YMax() >= frame.rows)
		{
			bb.SetYMin(frame.rows - bb.Height());
		}
		if (bb.XMin()<0)
		{
			bb.SetXMin(0);
		}
		if (bb.YMin()<0)
		{
			bb.SetYMin(0);
		}
		bkTrk->Init(frame, bb);
		mirrorList.push_back(bkTrk);

		// Refill seed trackers
		if(trkId%seedTrks.size()==seedTrks.size()-1) {
			seedTrks=GetNewTrackers();
		}
	}
	
	// Start tracing back
	for (int i = interval-1; i >=0 ; i--)
	{
		RectVec resList(trkN);
		const Mat& frame = frmList[i];

		if (i == interval - 1)
		{
			// Initialize last bounding box as first one
			for (int trkId = 0; trkId < trkN; trkId++)
			{
				resList[trkId] = trjList[trkId][i];
			}
		}
		else
		{
			#pragma omp parallel for
			for (int trkId = 0; trkId < trkN; trkId++)
			{
				ITrackerWrapper* tracker = mirrorList[trkId];
				resList[trkId] = tracker->Track(frame);
			}

		}
		for (int trkId = 0; trkId < trkN; trkId++)
		{
			trjBkList[trkId][i] = resList[trkId];
		}

		if (Display)
		{
			monitor.Display(frame, resList);
		}
	}
	for (int trackerId = 0; trackerId < mirrorList.size(); trackerId++)
	{
		delete mirrorList[trackerId];
	}

	return trjBkList;
}

vector<float> TrajectoriesRefiner::CalTrajectoryScores(
        const vector<RectVec> &trjList,
        const vector<RectVec> &trjBkList,
        const vector<Mat> &frmList
) const {
    assert(trjList.size() == trjBkList.size());
    assert(trjList.front().size() == frmList.size());

    size_t startFrame = 1, endFrame = trjList.at(0).size();
    size_t trackerAmount = trjList.size();
    size_t interval = endFrame - startFrame + 1;

    vector <vector<float>> vecBBDiffval;
    vecBBDiffval.resize(trackerAmount);

    vector<float> trajectoryScores;
    vector<vector<float>> vecOvl;

    trajectoryScores.resize(trackerAmount);

    for (int T_ID = 0; T_ID<trackerAmount; T_ID++)
    {
        vecBBDiffval[T_ID].resize(interval);
    }

    std::vector<std::vector<float>> vecBackCost;

    for (int T_ID = 0; T_ID < trackerAmount; T_ID++)
    {
        // Get image patches in bouding box from first frame
        vector<Mat> vecFrgImg;
        IntRect bb = trjList[T_ID][startFrame - 1];

        Mat roi = Mat((frmList)[startFrame - 1], cv::Rect(bb.XMin(), bb.YMin(), bb.Width(), bb.Height()));
        Mat tmpMat = roi.clone();
        vecFrgImg.push_back(tmpMat);

        vector<float> tmpOvrlCost;
        tmpOvrlCost.resize(interval);

        vector<float> tmpCost;
        tmpCost.resize(interval);
        trajectoryScores[T_ID] = 0.f;
        float tmpTRJDist = 0.f;

        for (int frmID = 0; frmID < interval; frmID++)
        {
            IntRect	currentBack = trjBkList.at(T_ID).at(frmID),
                    currentForward = trjList.at(T_ID).at(frmID);
            float x_diff = currentBack.XCentre() - currentForward.XCentre();
            float y_diff = currentBack.YCentre() - currentForward.YCentre();
            float ImgDiff = CalImgDiff(currentBack, vecFrgImg, (frmList[frmID]));	// Eq. 9
            float costVal = (float)(ImgDiff*expf(-(x_diff*x_diff + y_diff*y_diff) / 500.f));		// Eq. 10

            float foverlap = currentBack.Overlap(currentForward);

            tmpOvrlCost[frmID] = foverlap;

            tmpTRJDist += (costVal);
            tmpCost[frmID] = costVal;

        }
        trajectoryScores[T_ID] = tmpTRJDist;
        vecBackCost.push_back(tmpCost);

        vecOvl.push_back(tmpOvrlCost);
    }
    TimeDomainPattern(trajectoryScores, vecOvl, int(interval));

    return trajectoryScores;
}

int TrajectoriesRefiner::SelectTrajectory(
	const vector<RectVec>& trjList, 
	const vector<RectVec>& trjBkList, 
	const vector<Mat>& frmList
) const
{
    vector<float> trjScores= CalTrajectoryScores(trjList, trjBkList, frmList);
    size_t trackerAmount = trjList.size();

	int maxID = 0;
	float maxVal = FLT_MIN;
	for (int T_ID = 0; T_ID<trackerAmount; T_ID++)
	{
		if (maxVal<trjScores[T_ID])
		{
			maxVal = trjScores[T_ID];
			maxID = T_ID;
		}
	}
//	if (Delta==0)
//	{
//		occluded = false;
//	}
//	else
//	{ 
//		occluded=CheckFrameOcclusion(vecBackCost, trackerAmount, interval);
//	}
	return maxID;
}

vector<int>
TrajectoriesRefiner::SelectTrajectories(
        const vector<RectVec> &trjList,
        const vector<RectVec> &trjBkList,
        const vector<Mat> &frmList,
        const size_t tpN,
        int &bestTrjId
) const
{
    vector<float> trjScores= CalTrajectoryScores(trjList, trjBkList, frmList);
    size_t trackerAmount = trjList.size();

    vector<int> maxIDs;
    bestTrjId=0;
    for(int i=0;i<tpN;i++) {
        int maxID = i;
        for (int T_ID = i; T_ID < trackerAmount; T_ID+=tpN) {
            // Best trj over one specific type of tracker
            if (trjScores[maxID] < trjScores[T_ID]) {
                maxID = T_ID;
            }
            // Best trj over all trjs
            if (trjScores[bestTrjId] < trjScores[T_ID]) {
                bestTrjId=T_ID;
            }
        }
        maxIDs.push_back(maxID);
    }
//	if (Delta==0)
//	{
//		occluded = false;
//	}
//	else
//	{
//		occluded=CheckFrameOcclusion(vecBackCost, trackerAmount, interval);
//	}
    return maxIDs;
}

// Calculating appearance similarity
float TrajectoriesRefiner::CalImgDiff(
	const IntRect& inRect,								// Tracker's result
	const vector<Mat>& vecFrgImg,							// Image patches from bounding box
	const Mat& Images										// Raw images
) const
{
	int ntargetX = int(inRect.XMin());
	int ntargetY = int(inRect.YMin());

	double tmpSum = 0.0;
	// TODO: Get channel number scientifically
	int nChannel = Images.channels();

	IntRect bb = IntRect(inRect);
	Mat ImgDiffMask = GetImgDiffMask(bb.Height(), bb.Width());
	

	for (size_t imgID = 0; imgID<vecFrgImg.size(); imgID++)
	{
		float	scale_h = (float)inRect.Height() / vecFrgImg[imgID].rows,
				scale_w = (float)inRect.Width() / vecFrgImg[imgID].cols;
		for (int nh = 0; nh<bb.Height(); nh++)
		{
			for (int nw = 0; nw<bb.Width(); nw++)
			{
				for (int nc = 0; nc<nChannel; nc++)
				{
					int tmpDiff = (int)(vecFrgImg[imgID].at<Vec3b>((int)(floor(nh / scale_h)), (int)(floor(nw / scale_w)))[nc] - Images.at<Vec3b>(ntargetY + nh, ntargetX + nw)[nc]);
//                    tmpSum += ((double)(tmpDiff*tmpDiff)*ImgDiffMask.at<double>(nh, nw));
                    tmpSum += ((double)(tmpDiff*tmpDiff)*ImgDiffMask.at<double>(nh, nw));
				}
			}
		}
	}
	return (float)exp(-tmpSum / (double)(inRect.Width()*inRect.Height() * 900 * vecFrgImg.size()));		// Eq. 9
}

// Gaussian weight mask in Eq. 9 (symboled as K)
Mat TrajectoriesRefiner::GetImgDiffMask(int nHeight, int nWidth) const
{
	//	Mat m_ImgDiffMask(nHeight,nWidth,DataType<float>::type);
	Mat m_ImgDiffMask(nHeight, nWidth, DataType<double>::type);
	float fCenterX = (float)(nWidth - 1.f) / 2.f;
	float fCenterY = (float)(nHeight - 1.f) / 2.f;
	int nHeightSq = nHeight*nHeight;
	int nWidthSq = nWidth*nWidth;

	for (int h = 0; h<nHeight; h++)
	{
		float fhDiff = (float)(h - fCenterY);
		for (int w = 0; w<nWidth; w++)
		{
			float fwDiff = (float)(w - fCenterX);
			m_ImgDiffMask.at<double>(h, w) = exp(-(double)3.2*((double)(fhDiff*fhDiff) / nHeightSq + (double)(fwDiff*fwDiff) / nWidthSq));
		}
	}
	return m_ImgDiffMask;
}

// Eq. 8
void TrajectoriesRefiner::TimeDomainPattern(
	vector<float>& TrajectoryDist, 
	const vector<vector<float>>& vecOvl, 
	const int& frameInterval
) const
{
	int nFrmEnd = min(Epsilon, frameInterval);
	int nOccFlg = 0;
	size_t NumTracker = TrajectoryDist.size();
	for (size_t T_ID = 0; T_ID<NumTracker; T_ID++)
	{
		int nflg = 0;
		for (int frmID = 0; frmID <= nFrmEnd; frmID++)
		{
			if (vecOvl[T_ID][frmID] <= Zeta)
			{
				nflg++;
			}
		}
		// Change to check last frames
		for (int frmID = 0; frmID <= nFrmEnd; frmID++)
		{
			if (vecOvl[T_ID][vecOvl[T_ID].size() - frmID - 1] <= Zeta)
			{
				nflg++;
			}
		}
		if (nflg <= 1)
		{
			TrajectoryDist[T_ID] = TrajectoryDist[T_ID] * 1000000;	// Eq. 8
			nOccFlg++;
		}
	}
}

bool TrajectoriesRefiner::CheckFrameOcclusion(
	std::vector<std::vector<float>> vecBackCost,
	size_t trackerAmount,
	size_t interval
) const
{
	std::vector<int> vecOccFlag;
	vecOccFlag.resize(interval);

	for (int frmID = 0; frmID < interval; frmID++)
	{
		vecOccFlag[frmID] = 1;
		for (int T_ID = 0; T_ID<trackerAmount; T_ID++)
		{
			if (vecBackCost[T_ID][frmID]>Delta)
			{
				vecOccFlag[frmID] = 0;
				break;
			}
		}
	}

	bool isOcclusion = false;
	int ncounter = 0;
	for (int frmID = 0; frmID < interval; frmID++)
	{
		if (vecOccFlag[frmID] == 1)
		{
			ncounter++;
		}
		else
		{
			ncounter = 0;
		}
		if (ncounter >= DeltaDuration)
		{
			isOcclusion = true;
			break;
		}
	}

	return isOcclusion;
}

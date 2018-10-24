#pragma once
#include "Rect.h"
#include "Tracker/KCF/KcfWrapper.h"
#include "Tracker/Struck/StruckWrapper.h"
#include "Tracker/CT/CtWrapper.h"
#include "Tracker/DSST/include/DsstWrapper.h"

#include <string>



using namespace std;
class TaskConfig
{
public:
	void SetSeq(string name, string basePath, int startFrame, int endFrame, string seqZeroNum, string seqFormat, IntRect bbox,bool enableMonitor)
	{
		SeqName = name;
		SeqBasePath = basePath;
		SeqStartFrame = startFrame;
		SeqEndFrame = endFrame;
		SeqBbox = bbox;
		SeqZeroNum = seqZeroNum;
		SeqFormat = seqFormat;
		EnableMonitor = enableMonitor;
	};
	void SetSeq(char* argv[])
	{
		SetSeq(
			argv[1],
			argv[2],
			stoi(argv[3]),
			stoi(argv[4]),
			argv[5],
			argv[6],
			IntRect(stoi(argv[7]), stoi(argv[8]), stoi(argv[9]), stoi(argv[10])),
			stoi(argv[11])==1
		);
	}
	string GetFramePath(int FrameId) const
	{
		string SeqFullPath = SeqBasePath + SeqSubfix + "%0" +SeqZeroNum + "d." + SeqFormat;
		char imgPath[1024];
		sprintf(imgPath, SeqFullPath.c_str(), FrameId);
		return imgPath;
	};
	string GetResultOutputPath() const { return SeqName + ResultSubfix; };
	

	string	SeqName = "basketball",
		SeqBasePath = "E:\\ComputerVision2015\\benchmark\\dataset\\basketball\\img\\",
		SeqSubfix = "";
	int	SeqStartFrame = 1,
		SeqEndFrame = 725;
	IntRect SeqBbox = IntRect(198, 214, 34, 81);
	string	SeqZeroNum = "4",
			SeqFormat = "jpg";
	bool EnableMonitor = true;
	string ResultSubfix = "_result.txt";

	bool NormalizeSize = false;
	int NormalWidth = 320, NormalHeight = 240;
};

class MtsConfig: public TaskConfig
{
public:
	static ITrackerWrapper* _GetNewTracker()
	{
		KcfWrapper* hogKCF=new KcfWrapper;
		hogKCF->ConfigFeature(true,false,false);
		return hogKCF;
	}

	ITrackerWrapper* GetNewTracker()
	{
//		return new StruckWrapper;
//		return new TestTrackerWrapper;
//		return new SowpWrapper;
		TpN=1;
		return _GetNewTracker();
	}

	static vector<ITrackerWrapper*> _GetNewTrackers() {
		vector<ITrackerWrapper*> l;

		KcfWrapper* hogKCF=new KcfWrapper;
		hogKCF->ConfigFeature(true,false,false);

//		KcfWrapper* labKCF=new KcfWrapper;
//		labKCF->ConfigFeature(true,true,false);

//		KcfWrapper* hisKCF = new KcfWrapper;
//		hisKCF->ConfigFeature(true, false, true);

//		ITrackerWrapper* struck = new StruckWrapper;
//		ITrackerWrapper* ct = new CtWrapper;
//		ITrackerWrapper* DSST = new DsstWrapper;

		l.push_back(hogKCF);
//		l.push_back(labKCF);
//		l.push_back(hisKCF);

//		l.push_back(struck);
//		l.push_back(ct);
//		l.push_back(DSST);

		return l;
	}

	vector<ITrackerWrapper*> GetNewTrackers()
	{
		vector<ITrackerWrapper*> l=_GetNewTrackers();
		TpN=l.size();

		return l;
	}

	size_t GetSnapshotInterval() const
	{
		return snapshotInterval;
	}

	void SetSnapshotInterval(size_t t_SnapshotInterval)
	{
		snapshotInterval = t_SnapshotInterval;
	}

	size_t GetSnapshotN() const
	{
		return snapshotN;
	}

	void SetSnapshotN(size_t t_SnapshotN)
	{
		snapshotN = t_SnapshotN;
	}

	int GetEpsilon() const
	{
		return Epsilon;
	}

	void SetEpsilon(int t_Epsilon)
	{
		Epsilon = t_Epsilon;
	}

	float GetZeta() const
	{
		return Zeta;
	}

	void SetZeta(float t_Zeta)
	{
		Zeta = t_Zeta;
	}

	float GetDelta() const
	{
		return Delta;
	}

	void SetDelta(float t_Delta)
	{
		Delta = t_Delta;
	}

	int GetDeltaDuration() const
	{
		return DeltaDuration;
	}

	void SetDeltaDuration(int t_DeltaDuration)
	{
		DeltaDuration = t_DeltaDuration;
	}

	bool GetPreservedOne() const
	{
		return PreservedOne;
	}

	void SetPreservedOne(bool t_PreservedOne)
	{
		PreservedOne = t_PreservedOne;
	}

	size_t GetTpN() {
		return TpN;
	}
private:
	size_t snapshotInterval = 10;
	size_t snapshotN = 6;
	int Epsilon = 4;
	float Zeta = 0.3f;
	float Delta = 0.004f;
	int DeltaDuration = 20;
	size_t TpN=0;					// Amount ot tracker types

	bool PreservedOne = false;
};

// Tracking.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#if defined(WIN32) || defined(_WIN32)
#include <io.h>
#else
#include <dirent.h>
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <armadillo>

using namespace std;
using namespace cv;

// Read file from directory
static void readDirectory(const string& directoryName, vector<string>& filenames, const string& type, bool addDirectoryName = true)
{
	filenames.clear();

#if defined(WIN32) | defined(_WIN32)
	struct _finddata_t s_file;
	string str = directoryName + "\\*." + type;

	intptr_t h_file = _findfirst(str.c_str(), &s_file);
	if (h_file != static_cast<intptr_t>(-1.0))
	{
		do
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "\\" + s_file.name);
			else
				filenames.push_back((string)s_file.name);
		} while (_findnext(h_file, &s_file) == 0);
	}
	_findclose(h_file);
#else
	DIR* dir = opendir(directoryName.c_str());
	if (dir != NULL)
	{
		struct dirent* dent;
		while ((dent = readdir(dir)) != NULL)
		{
			if (addDirectoryName)
				filenames.push_back(directoryName + "/" + string(dent->d_name));
			else
				filenames.push_back(string(dent->d_name));
		}

		closedir(dir);
	}
#endif

	sort(filenames.begin(), filenames.end());

}

// Read groundtruth from txtfile
static int readRect(const string& fileName, vector<Rect>& groundtruth)
{
	int retVal = 0;
	vector<int> position;
	Rect temp;
	int idata;
	int count = 0;
	ifstream inFile(fileName);
	if (!inFile.is_open())
	{
		cout << "Read groundtruth failed..." << endl;
		retVal = -1;
		return(retVal);
	}

	while (inFile)
	{
		inFile >> idata;
		position.push_back(idata);
	}

	for (int i = 0; i < position.size()  / 4; ++i)
	{
		temp.x = position[4 * i + 0];
		temp.y = position[4 * i + 1];
		temp.width = position[4 * i + 2];
		temp.height = position[4 * i + 3];
		groundtruth.push_back(temp);
	}

	vector<int>(position).swap(position);

	return 1;
}

struct opt
{
	double condenssig = 0.05;
	bool useNormalsize = false;

	// Adjust as needed if high resolution image is used
	int normalWidth = 320;
	int normalHeight = 240;

	//int FeatureExtractor_tmplsize[] = { 32, 32 };
	int FeatureExtractor_NumBins = 8;

	int HaarExtractor_M = 512;
	int HaarExtractor_maxNumRect = 6;
	int HaarExtractor_minNumRect = 2;

	int Sampler_NegSlidingWindowSampler_NegSlidingH = 100;
	int Sampler_NegSlidingWindowSampler_NegSlidingW = 100;
	int Sampler_NegSlidingWindowSampler_NegStride = 5;
	double Sampler_NegSlidingWindowSampler_excludeNegRatio = 0.3;

	int Sampler_PosSlidingWindowSampler_PosSlidingH = 5;
	int Sampler_PosSlidingWindowSampler_PosSlidingW = 5;

	int MotionModel_SlidingWindowMotionModel_slidingH = 30;
	int MotionModel_SlidingWindowMotionModel_slidingW = 30;
	int MotionModel_SlidingWindowMotionModel_stride = 2;

	int MotionModel_RadiusSlidingWindowMotionModel_radius = 30;
	int MotionModel_RadiusSlidingWindowMotionModel_stride = 2;

	int MotionModel_ParticleFilterMotionModel_N = 400;
	//static const double MotionModel_ParticleFilterMotionModel_affsig[4] = { 6, 6, 0.01, 0.001 };

	double ClassificationScoreJudger_thresold = 0.9;

	bool useFirstFrame = false;

	int SOSVM_kernel = 0;

	string KernelRidge_ker = "rbf";
	int KernelRidge_RegulationTerm = 1;
	int KernelRidge_rbfSigma = 1;
	//int KernelRidge_polyParmeters[] = { 1, 2 };

}opt;

// help function
// *********************meshgrid*********************
/*
static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

// helper function (maybe that goes somehow easier)
static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat1i &X, cv::Mat1i &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}*/
// *********************meshgrid*********************

// ********************* cumsum *********************
Mat1d cumsum(Mat1d& src, int rc = 1)
{
	Mat1d dst;
	src.copyTo(dst);
	if (rc == 1)
	{
		for (int i = 1; i < dst.rows; ++i)
		{
			for (int j = 0; j < dst.cols; ++j)
				dst.at<double>(i, j) += dst.at<double>(i - 1, j);
		}
	}
	if (rc == 2)
	{
		for (int j = 1; j < dst.cols; ++j)
		{
			for (int i = 0; i < dst.rows; ++i)
				dst.at<double>(i, j) += dst.at<double>(i, j - 1);
		}
	}
	return dst;
}
// ********************* cumsum *********************
// Sampler
// PosSlidingWindowsSampler_opencv
/*
void PosSlidingWindowSampler(vector<double>& initTmpl, Mat1d& tmplpos)
{
	int slidingH = 5;
	int slidingW = 5;

	vector<int> hVec, wVec;
	for (int i = 0; i < slidingH + 1; ++i)
		hVec.push_back(i - (int)round((double)slidingH / 2.0));
	for (int i = 0; i < slidingW + 1; ++i)
		wVec.push_back(i - (int)round((double)slidingW / 2.0));
	Mat hMat, wMat;
	meshgrid(Mat(wVec), Mat(hVec), wMat, hMat);
	int num = (slidingH + 1) * (slidingW + 1);
	Mat tmpl;
	repeat(Mat(initTmpl).t(), num, 1, tmpl);
	Mat1d hMat_colvec, wMat_colvec;
	Mat1d hMat_t, wMat_t;
	hMat_t = hMat.t();
	wMat_t = wMat.t();
	hMat_colvec = hMat_t.reshape(1, 1).t();
	wMat_colvec = wMat_t.reshape(1, 1).t();
	tmpl.col(0) = tmpl.col(0) + wMat_colvec;
	tmpl.col(1) = tmpl.col(1) + hMat_colvec;

	tmplpos = Mat(initTmpl).t();
	tmplpos.push_back(tmpl);
}*/
void meshgrid(arma::imat& wVec, arma::imat& hVec, arma::imat& wMat, arma::imat& hMat)
{
	wMat = arma::repmat(wVec, hVec.size(), 1);
	hMat = arma::repmat(hVec.t(), 1, wVec.size());
}
void PosSlidingWindowSampler(arma::mat& initTmpl, arma::mat& tmplpos)
{
	int slidingH = 5;
	int slidingW = 5;

	vector<int> hVec, wVec;
	for (int i = 0; i < slidingH + 1; ++i)
		hVec.push_back(i - (int)round((double)slidingH / 2.0));
	for (int i = 0; i < slidingW + 1; ++i)
		wVec.push_back(i - (int)round((double)slidingW / 2.0));

	arma::imat hvec(hVec), wvec(wVec);
	hvec = hvec.t();
	wvec = wvec.t();
	arma::imat hMat, wMat;
	int num = hVec.size() * wVec.size();
	meshgrid(wvec, hvec, wMat, hMat);
	tmplpos = arma::repmat(initTmpl, num, 1);
	tmplpos.col(0) = tmplpos.col(0) + arma::reshape(wMat,num,1);
	tmplpos.col(1) = tmplpos.col(1) + arma::reshape(hMat, num, 1);
	tmplpos.insert_rows(0, initTmpl);
}
void NegSlidingWindowSampler(arma::mat& initTmpl, arma::mat& tmplneg)
{
	int slidingH = 100;
	int slidingW = 100;
	int stride = 5;
	double ratio = 0.3;

	vector<int> hVec, wVec;
	for (int i = 0; i < slidingH + 1; i = i + stride)
		hVec.push_back(i - (int)round((double)slidingH / 2.0));
	for (int i = 0; i < slidingW + 1; i = i + stride)
		wVec.push_back(i - (int)round((double)slidingW / 2.0));
	arma::imat hvec(hVec), wvec(wVec);
	hvec = hvec.t();
	wvec = wvec.t();
	arma::imat hMat, wMat;
	int num = hVec.size() * wVec.size();
	meshgrid(wvec, hvec, wMat, hMat);
	tmplneg.zeros(num*initTmpl.n_rows, 4);
	arma::mat idx;
	idx.zeros(num*initTmpl.n_rows, 1);
	arma::mat tempTmpl;
	for (int i = 0; i < initTmpl.n_rows; ++i)
	{
		tempTmpl = repmat(initTmpl.row(i), num, 1);
		tempTmpl.col(0) = tempTmpl.col(0) + arma::reshape(wMat, num, 1);
		tempTmpl.col(1) = tempTmpl.col(1) + arma::reshape(hMat, num, 1);

	}
}
// Motion
// ParticleFilterMotionModel
/*
void ParticleFilterMotionModel(vector<double>& initTmpl, Mat1d& initConf, Mat1d& tmpl)
{
	double affsig_array[4] = { 6, 6, 0.01, 0.001 };
	vector<double> affsig_vec(affsig_array, affsig_array + 4);
	int N = 400;
	int szH = 32;

	double minVal, maxVal;
	int minIdx, maxIdx;
	minMaxIdx(initConf, &minVal, &maxVal);
	Mat1d minVal_mat(initConf.rows, initConf.cols, -minVal);
	add(initConf, minVal_mat, initConf);
	Mat1d condenssig(initConf.rows, initConf.cols, 0.05);
	initConf = initConf / condenssig;
	initConf = initConf.t();
	exp(initConf, initConf);
	Mat1d initConf_sum(initConf.rows, initConf.cols, sum(initConf).val[0]);
	initConf = initConf / initConf_sum;
	minMaxIdx(initConf, &minVal, &maxVal, &minIdx, &maxIdx);

	Mat1d initTmpl_mat(initTmpl);
	initTmpl_mat.t();
	if (initTmpl_mat.rows == 1)
		repeat(initTmpl, N, 1, initTmpl);
	else
		N = initTmpl_mat.rows;
		Mat1d cumconf = cumsum(initConf,1);
		Mat1d rand(1,N);
		randu(rand, 0, 1);



		Mat1d minVal_;
}*/
void ParticleFilterMotionModel(arma::mat& initTmpl, arma::mat& initConf, arma::mat& tmpl)
{
	double affsig_array[4] = { 6, 6, 0.01, 0.001 };
	vector <double> affsig_vec(affsig_array, affsig_array + 4);
	int N = 400;
	int szH = 32;
	double condenssig = 0.05;

	initConf = initConf - min(initConf);
	initConf = arma::exp(initConf / condenssig).t();
	arma::mat sum_initConf_mat = arma::sum(initConf);
	double sum_initConf = sum_initConf_mat.at(0,0);
	initConf = initConf / sum_initConf;
	arma::uword index;
	double max_value = initConf.min(index);

	if (initTmpl.n_rows == 1)
		tmpl = arma::repmat(initTmpl, N, 1);
	else
	{
		N = initTmpl.n_rows;
		arma::mat cumconf = arma::cumsum(initConf,0);
		arma::mat rnd = arma::randu(1, N);
		rnd = arma::repmat(rnd, N, 1);
		cumconf = arma::repmat(cumconf, 1, N);
		arma::umat compare = (rnd > cumconf);
		compare = arma::sum(compare);
		arma::umat idx = floor(compare) + 1;
	}

	tmpl.col(3) = tmpl.col(3) / tmpl.col(2);
	tmpl.col(2) = tmpl.col(2) / szH;
	arma::mat rnd2 = arma::randn(N, 4);
	arma::mat valTmpl(affsig_vec);
	valTmpl = valTmpl.t();
	tmpl = tmpl + rnd2 % repmat(valTmpl, N, 1);
	tmpl.col(2) = tmpl.col(2) * szH;
	tmpl.col(3) = tmpl.col(2) % tmpl.col(3);

	vector<unsigned> rndIdx_vec,colIdx_vec;
	for (int i = 0; i < tmpl.n_rows; ++i)
		rndIdx_vec.push_back(i);
	for (int i = 0; i < tmpl.n_cols; ++i)
		colIdx_vec.push_back(i);
	arma::uvec rndIdx(rndIdx_vec);
	arma::uvec colIdx(colIdx_vec);
	rndIdx = arma::shuffle(rndIdx);
	tmpl = tmpl(rndIdx,colIdx);
	tmpl = tmpl.submat(0, 0, tmpl.n_rows - 2, tmpl.n_cols - 1);
	tmpl.insert_rows(0, initTmpl);
	tmpl = tmpl(find(tmpl.col(2) > 3),colIdx);
	tmpl = tmpl(find(tmpl.col(3) > 3),colIdx);
}

void tracking(vector<string>& images_filenames, Rect& targetIni, bool saveImage = false)
{
	Mat frame;
	//frame = imread(images_filenames[0], 1);
	//cvtColor(frame, frame,CV_RGB2GRAY);
	double p[4] = { targetIni.x + (double)targetIni.width / 2,
		         targetIni.y + (double)targetIni.height / 2,
		         targetIni.width,
		         targetIni.height };
	vector<double> pvec(p, p + 4);
	arma::mat pvec_mat(pvec);
	pvec_mat = pvec_mat.t();
	arma::mat tmplPos, tmplNeg, tmpl,prob;

	for (int i = 0; i < images_filenames.size(); ++i)
	{
		frame = imread(images_filenames[i], 1);
		frame.convertTo(frame, CV_32FC3);

		if (i != 0)
		{
			ParticleFilterMotionModel(tmpl, prob, tmpl);

		}
		else
		{
			arma::mat initConf;
			initConf.ones(1, 1);
			ParticleFilterMotionModel(pvec_mat, initConf, tmpl);
			prob.ones(1, tmpl.n_rows);
		}

		if (i == 0)
		{
			PosSlidingWindowSampler(pvec_mat, tmplPos);
			NegSlidingWindowSampler(pvec_mat, tmplNeg);
		}

	}
}
int _tmain(int argc, _TCHAR* argv[])
{
	string name = "Surfer";
	string datapath = "trackingDataset\\" + name;
	string imagespath = datapath + "\\img";
	string groundtruthpath = datapath + "\\groundtruth_rect.txt";
	vector<string> images_filenames;
	vector<Rect> groundtruth;
	string type = "jpg";

	cout << "Loading data..." << endl;
	readDirectory(imagespath, images_filenames, type , true);

	if (readRect(groundtruthpath, groundtruth))
		cout << "Loading data done..." << endl;
	else
		cout << "Loading data failed..." << endl;

	Rect targetIni = groundtruth[0];
	Mat img;

	tracking(images_filenames, targetIni, false);

	for (int i = 0; i < groundtruth.size(); i++)
	{
		img = imread(images_filenames[i], 1);
		rectangle(img, groundtruth[i], Scalar(0, 0, 255),4,8,0);
		imshow("img", img);
		waitKey(20);
	}
	

	waitKey(0);
	return 0;
}



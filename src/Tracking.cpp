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

	Mat img;

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



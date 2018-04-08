#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\ml\ml.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

using namespace cv;
using namespace std;
using namespace cv::ml;

#define PosNum 2416
#define NegNum 15873
#define TRAIN false
#define FIND false
int main()
{
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	Ptr<SVM> svm = SVM::create();
	if (TRAIN)
	{
		string ImgPath;
		ifstream finPos("D:\\VS2015project\\data\\pos\\pos.txt");
		ifstream finNeg("D:\\VS2015project\\data\\neg\\neg.txt");

		if (!finPos||!finNeg)
		{
			cout << "Pos/Neg reading failed..." << endl;
			return 1;
		}
		Mat sampleFeatureMat;
		Mat sampleLabelMat;

		//loading original positive examples...
		for (int i = 0; i < PosNum&&getline(finPos,ImgPath); i++)
		{
			cout << ImgPath << ":procesing..." << endl;
			Mat src = imread(ImgPath);
			src = src(Rect(16, 16, 64, 128));//Resize
			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8),Size(0,0));
			if (i==0)
			{
				sampleFeatureMat = Mat::zeros(PosNum + NegNum, descriptors.size(), CV_32FC1);
				sampleLabelMat = Mat::zeros(PosNum + NegNum, 1, CV_32SC1);
			}
			for (size_t j = 0; j < descriptors.size(); j++)
			{
				sampleFeatureMat.at<float>(i, j) = descriptors[j];
			}
			sampleLabelMat.at<int>(i, 0) = 1;//'1'表示有人
		}
		finPos.close();
		cout << "Extract PosSampleFeature Done..." << endl;

		//loading negative examples...
		for (int i = 0; i < NegNum&&getline(finNeg, ImgPath); i++)
		{
			cout << ImgPath << ":procesing..." << endl;
			Mat src = imread(ImgPath);
			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8),Size(0,0));
			for (size_t j = 0; j < descriptors.size(); j++)
			{
				sampleFeatureMat.at<float>(i+PosNum, j) = descriptors[j];
			}
			sampleLabelMat.at<int>(PosNum+i, 0) = -1;
		}
		finNeg.close();
		cout << "Extract NgeSampleFeature Done..." << endl;

		//配置SVM
		svm->setType(SVM::C_SVC);//可以尝试一下one class
		svm->setC(0.01);
		svm->setKernel(SVM::LINEAR);//必须是线性的
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON));
		Ptr<TrainData> tData = TrainData::create(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
		cout << "Starting Training..." << endl;
		svm->train(tData);
		cout << "Finishing Training..." << endl;
		svm->save("peopledetect.xml");
	}
	else
	{
		svm = StatModel::load<SVM>("peopledetect.xml");
		cout << "loding complete..." << endl;
	}

	Mat sv = svm->getSupportVectors();
	int sv_total = sv.rows;
	int svdim = svm->getVarCount();
	cout << "支持向量个数：" << sv_total << endl;
	Mat alpha = Mat::zeros(sv_total, svdim, CV_32FC1);
	Mat svidx = Mat::zeros(1, sv_total, CV_64F);
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	alpha.convertTo(alpha, CV_32FC1);
	Mat result;
	result = -1 * alpha*sv;
	vector<float> hog_detector;
	for (int i = 0; i < svdim; i++)
	{
		hog_detector.push_back(result.at<float>(0, i));
	}
	hog_detector.push_back(rho);
	ofstream fout("peopleDetector.txt");
	for (size_t i = 0; i < hog_detector.size(); i++)
	{
		fout << hog_detector[i] << endl;
	}
	//over....
	//testting...then find hard examples
	hog.setSVMDetector(hog_detector);


	if (FIND)
	{
		ifstream finONeg("F:\\INRIADATA\\normalized_images\\train\\neg\\neg.txt");
		string HardImgPath;
		int Hcount=0;
		while (getline(finONeg,HardImgPath))
		{
			cout << HardImgPath << "processing..." << endl;
			Mat src = imread(HardImgPath);
			vector<Rect> found;
			hog.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			for (size_t i = 0; i < found.size(); i++)
			{
				Rect r = found[i];
				if (r.x< 0)
					r.x = 0;
				if (r.y<0)
					r.y = 0;
				if (r.x + r.width>src.cols)
					r.width = src.cols - r.x;
				if (r.y + r.height>src.rows)
					r.height = src.rows - r.y;
				Mat hardExampleImg = src(r);
				resize(hardExampleImg, hardExampleImg, Size(64, 128));
				string saveName = to_string(++Hcount) + ".png";
				imwrite(saveName, hardExampleImg);
			}
		}
	}

	else
	{
		Mat src_test = imread("test3.png");
		vector<Rect> foundRect, foundRect_filtered;
		hog.detectMultiScale(src_test, foundRect, 0, Size(2, 2), Size(32, 32), 1.03, 2, false);
		cout << foundRect.size() << endl;
		for (size_t i = 0; i < foundRect.size(); i++)
		{
			Rect r = foundRect[i];
			int j = 0;
			for (; j < foundRect.size(); j++)
				if (j != i && (r&foundRect[j]) == r)
					break;
			if (j == foundRect.size())
				foundRect_filtered.push_back(r);
		}
		cout << foundRect_filtered.size() << endl;
		for (size_t i = 0; i < foundRect_filtered.size(); i++)
		{
			Rect r = foundRect_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src_test, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		}
		imshow("result", src_test);
		imwrite("ff.png", src_test);
	}
	waitKey(0);
	return 0;
}
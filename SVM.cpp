#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "CSVparser.hpp"

#pragma warning(disable: 4996)

using namespace std;
using namespace cv;
using namespace cv::ml;

 //#define train_data_count 2240
 //#define test_data_count 560

#define train_data_count 2000
#define test_data_count 840


int main(int, char**)
{
	// Parser csv
	csv::Parser Parser_train_data = csv::Parser("files/train.csv");
	csv::Parser Parser_test_data = csv::Parser("files/test.csv");

	// 4 feather
	int train_labels[train_data_count];
	float training_Data[train_data_count][4];

	int test_labels[test_data_count];
	float test_data[test_data_count][4];
	int test_repone[test_data_count];

	// csv -> array
	for (int i = 0; i < train_data_count; i++)
	{
		train_labels[i] = atoi((Parser_train_data[i][0]).c_str());
		for (int j = 0; j <= 3; j++)
		{
			training_Data[i][j] = strtof((Parser_train_data[i][j + 1]).c_str(), 0);
		}
	}


	for (int i = 0; i < test_data_count; i++)
	{
		test_labels[i] = atoi((Parser_test_data[i][0]).c_str());

		for (int j = 0; j <= 3; j++)
		{
			test_data[i][j] = strtof((Parser_test_data[i][j + 1]).c_str(), 0);
		}
	}

	// array -> Mat
	Mat training_data_Mat(train_data_count, 4, CV_32F, training_Data);
	Mat labels_Mat(train_data_count, 1, CV_32SC1, train_labels);
	Mat test_data_Mat(test_data_count, 4, CV_32F, test_data);
	Mat test_repone_Mat(test_data_count, 1, CV_32SC1, test_repone);


	// autoTrain the SVM
	Ptr<SVM> svm = SVM::create();
	Ptr<TrainData> Autotrain_parameter = TrainData::create(training_data_Mat, ROW_SAMPLE, labels_Mat);
	svm->trainAuto(Autotrain_parameter);


	// svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	// svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	cout << "Finished training process" << endl;

	svm->save("svm.xml");
		svm->predict(test_data_Mat, test_repone_Mat);

	// Accuracy
	float count = 0;
	float accuracy = 0;

	for (int i = 0; i < test_repone_Mat.rows; i++)
	{
		//cout << i <<"  "<< test_repone_mat.at<float>(i, 0) << "   " << test_labels[i] << endl;
		if (test_repone_Mat.at<float>(i, 0) == test_labels[i])
			count = count + 1;
	}

	accuracy = (count / test_repone_Mat.rows) * 100;
	cout << count << "/" << test_repone_Mat.rows << endl;

	cout << accuracy;

	return 0;
}

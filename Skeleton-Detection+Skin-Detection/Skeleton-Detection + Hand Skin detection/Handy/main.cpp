#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "BackgroundRemover.h"
#include "SkinDetector.h"
#include "FaceDetector.h"
#include "FingerCount.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const int POSE_PAIRS[20][2] =
{
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // index
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
};

int BPOSE_PAIRS[17][2] =
{
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
};
int nPoints2 = 18;

string protoFile = "hand/pose_deploy.prototxt";
string weightsFile = "hand/pose_iter_102000.caffemodel";

string protoFile2 = "pose/coco/pose_deploy_linevec.prototxt";
string weightsFile2 = "pose/coco/pose_iter_440000.caffemodel";

int nPoints = 22;

int main(int argc, char** argv)
{
    Mat frame, frameOut, handMask, foreground, fingerCountDebug;

	BackgroundRemover backgroundRemover;
	SkinDetector skinDetector;
	FaceDetector faceDetector;
	FingerCount fingerCount;

    float thresh = 0.7;

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }

  //  Mat frame, 
    Mat frameCopy;

    cap.set(CAP_PROP_FRAME_HEIGHT, 216);
    cap.set(CAP_PROP_FRAME_WIDTH, 216);

    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    float aspect_ratio = frameWidth / (float)frameHeight;
    int inHeight = 368;
    int inWidth = (int(aspect_ratio * inHeight) * 8) / 8;

 //   cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

    VideoWriter video("Output-Skeleton.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frameWidth, frameHeight));

    Net net = readNetFromCaffe(protoFile, weightsFile);

    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
/*
    Net net2 = readNetFromCaffe(protoFile2, weightsFile2);

    net2.setPreferableBackend(DNN_BACKEND_CUDA);
    net2.setPreferableTarget(DNN_TARGET_CUDA);
*/
    double t = 0;
    while (1)
    {
        cap >> frame;
        frameOut = frame.clone();
        //d_src.download(frame);
        skinDetector.drawSkinColorSampler(frameOut);

        foreground = backgroundRemover.getForeground(frame);

        faceDetector.removeFaces(frame, foreground);
        handMask = skinDetector.getSkinMask(foreground);
        fingerCountDebug = fingerCount.findFingersCount(handMask, frameOut);
        Mat frame2, finger2;
        //---------------
        double t = (double)cv::getTickCount();

        cap >> frame;
        frameCopy = frame.clone();
        Mat inpBlob = blobFromImage(frame, 1.0 / 255, Size(frameWidth, frameHeight), Scalar(0, 0, 0), false, false);

        net.setInput(inpBlob);

        Mat output = net.forward();

        int H = output.size[2];
        int W = output.size[3];
/*
        Mat inpBlob2 = blobFromImage(frame, 1.0 / 255, Size(frameWidth, frameHeight), Scalar(0, 0, 0), false, false);

        net2.setInput(inpBlob2);

        Mat output2 = net2.forward();

        int H2 = output2.size[2];
        int W2 = output2.size[3];



        //-------------------------------------

        vector<Point> points2(nPoints2);
        for (int n = 0; n < nPoints2; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap(H, W, CV_32F, output2.ptr(0, n));

            Point2f p(-1, -1);
            Point maxLoc;
            double prob;
            minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
            if (prob > thresh)
            {
                p = maxLoc;
                p.x *= (float)frameWidth / W;
                p.y *= (float)frameHeight / H;

                circle(frameCopy, cv::Point((int)p.x, (int)p.y), 3, Scalar(0, 255, 255), -1);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1.1, cv::Scalar(0, 0, 255), 2);
            }
            points2[n] = p;
        }

        int nPairs2 = sizeof(BPOSE_PAIRS) / sizeof(BPOSE_PAIRS[0]);

        for (int n = 0; n < nPairs2; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f partA = points2[BPOSE_PAIRS[n][0]];
            Point2f partB = points2[BPOSE_PAIRS[n][1]];

            if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
                continue;

            line(frame, partA, partB, Scalar(0, 255, 255), 8);
            circle(frame, partA, 3, Scalar(0, 0, 255), -1);
            circle(frame, partB, 3, Scalar(0, 0, 255), -1);
        }

        //-------------------------------------
*/

        // find the position of the body parts
        vector<Point> points(nPoints);
        for (int n = 0; n < nPoints; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap(H, W, CV_32F, output.ptr(0, n));
            resize(probMap, probMap, Size(frameWidth, frameHeight));

            Point maxLoc;
            double prob;
            minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
            if (prob > thresh)
            {
                circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 3, Scalar(0, 255, 255), -1);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            }
            points[n] = maxLoc;
        }

        int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

        for (int n = 0; n < nPairs; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f partA = points[POSE_PAIRS[n][0]];
            Point2f partB = points[POSE_PAIRS[n][1]];

            if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
                continue;

            line(frame, partA, partB, Scalar(0, 255, 255), 8);
            circle(frame, partA, 3, Scalar(0, 0, 255), -1);
            circle(frame, partB, 3, Scalar(0, 0, 255), -1);
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Time Taken for frame = " << t << endl;
        cv::putText(frame, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
        // imshow("Output-Keypoints", frameCopy);

        //frame2 = frame.clone();
        resize(frame, frame2, Size(256, 256), 0, 0, INTER_LINEAR);
        resize(fingerCountDebug, finger2, Size(frame2.size().width,frame2.size().height), 0, 0, INTER_LINEAR);

        for (int i = 0; i < frame2.size().width; i++) {
            for (int j = 0; j < frame2.size().height; j++) {
                if (finger2.at<Vec3b>(i, j)[0] != 0 || finger2.at<Vec3b>(i, j)[1] != 0
                    || finger2.at<Vec3b>(i, j)[2] != 0) {
                    frame2.at<Vec3b>(i, j)[0] = finger2.at<Vec3b>(i, j)[0];
                    frame2.at<Vec3b>(i, j)[1] = finger2.at<Vec3b>(i, j)[1];
                    frame2.at<Vec3b>(i, j)[2] = finger2.at<Vec3b>(i, j)[2];
                }
            }
         }
        video.write(frame2);
        resize(frame2, frame2, Size(512, 512), 0, 0, INTER_LINEAR);
        resize(frame, frame, Size(512, 512), 0, 0, INTER_LINEAR);
        resize(fingerCountDebug, fingerCountDebug, Size(512, 512), 0, 0, INTER_LINEAR);
        imshow("output", frame);
        imshow("handDetection", fingerCountDebug);
        imshow("test", frame2);

   //     video.write(frame2);

        int key = waitKey(1);

        if (key == 27) // esc
            break;
        else if (key == 98 || key == 66) // b
            backgroundRemover.calibrate(frame);
        else if (key == 115 || key == 81) // s
            skinDetector.calibrate(frame);

   //     imshow("Output-Skeleton", frame);
    }
    // When everything done, release the video capture and write object
    cap.release();
    video.release();

    return 0;
}

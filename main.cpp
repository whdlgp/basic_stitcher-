#include "stitcher.hpp"

#include <thread>
#include <future>
#include "atomicops.h"
#include "readerwriterqueue.h"

using namespace std;
using namespace cv;
using namespace moodycamel;

#define QUEUE_SIZE 500

class thread_args
{
    public :
    vector<Mat> imgs;
};

class thread_output
{
    public :
    Mat pano;
};

BlockingReaderWriterQueue<thread_args> th_arg0(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg1(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg2(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg3(QUEUE_SIZE);

BlockingReaderWriterQueue<thread_output> th_out0(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out1(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out2(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out3(QUEUE_SIZE);

void stitcher_thread(int idx)
{
    while(true)
    {
        thread_args th_arg;

        switch(idx)
        {
            case 0:
            th_arg0.wait_dequeue(th_arg);
            break;

            case 1:
            th_arg1.wait_dequeue(th_arg);
            break;

            case 2:
            th_arg2.wait_dequeue(th_arg);
            break;

            case 3:
            th_arg3.wait_dequeue(th_arg);
            break;
        }

        thread_output th_out;
        Basic_stitcher stitcher(false);
        th_out.pano =stitcher.stitcher_do_all(th_arg.imgs);
        
        switch(idx)
        {
            case 0:
            th_out0.enqueue(th_out);
            break;

            case 1:
            th_out1.enqueue(th_out);
            break;
            
            case 2:
            th_out2.enqueue(th_out);
            break;
            
            case 3:
            th_out3.enqueue(th_out);
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    STICHER_DBG_OUT("start");
    VideoCapture vid0("videofile0.avi");
    VideoCapture vid1("videofile1.avi");
    VideoCapture vid2("videofile2.avi");

    vector<Mat> output;

    thread thread0(stitcher_thread, 0);
    thread thread1(stitcher_thread, 1);
    thread thread2(stitcher_thread, 2);
    thread thread3(stitcher_thread, 3);
    
    {
        vector<Mat> vids(3);
        vid0 >> vids[0];
        vid1 >> vids[1];
        vid2 >> vids[2];

        STICHER_DBG_OUT("start stitching first frame");
        Basic_stitcher stitcher(false);
        Mat pano = stitcher.stitcher_do_all(vids);
        STICHER_DBG_OUT("push to output queue");
        output.push_back(pano);
    }

    int capture_count = 0;
    while(capture_count < 12)
    {
        STICHER_DBG_OUT("Put frame");
        
        vector<Mat> vids(3);
        
        thread_args th_arg;
        Mat tmp;
        vid0 >> tmp;
        th_arg.imgs.push_back(tmp.clone());
        vid1 >> tmp;
        th_arg.imgs.push_back(tmp.clone());
        vid2 >> tmp;
        th_arg.imgs.push_back(tmp.clone());

        switch(capture_count % 4)
        {
            case 0:
            th_arg0.enqueue(th_arg);
            break;

            case 1:
            th_arg1.enqueue(th_arg);
            break;
            
            case 2:
            th_arg2.enqueue(th_arg);
            break;
            
            case 3:
            th_arg3.enqueue(th_arg);
            break;
        }

        capture_count++;
    }

    for(int i = 0; i < capture_count; i++)
    {
        thread_output th_out;
        switch(i % 4)
        {
            case 0:
            th_out0.wait_dequeue(th_out);
            break;

            case 1:
            th_out1.wait_dequeue(th_out);
            break;
            
            case 2:
            th_out2.wait_dequeue(th_out);
            break;
            
            case 3:
            th_out3.wait_dequeue(th_out);
            break;
        }

        output.push_back(th_out.pano);
    }

    for(int i = 0; i < output.size(); i++)
    {
        Mat result;
        output[i].convertTo(result, CV_8UC1);
        imshow("stitch output", result);
        waitKey(0);
    }

    STICHER_DBG_OUT("stitching completed successfully\n");
    return 0;
}
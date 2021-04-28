#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio/legacy/constants_c.h>
#include "utils.hpp"
#include "pose_post_process.hpp"


#define DUMP_OUTDATA 0
#define SHOW_FROME_FILE 0
static int framecnt = 0;
void paint_image_single(cv::Mat &ori_img, std::vector<float> &predictions);
int post_process_1(
        long inputs[],
        int net_n,
        int origin_imgw,
        int origin_imgh,
        int net_inw,
        int net_inh,
        int net_outw,
        int net_outh,
        int input_tensor_num,
        int out_size,
        unsigned char *ori_image)
{
    float netoutdata[out_size];
    float scale_to_float =0.0099951f;
    float nms_scale;
    int newh = net_inh;
    float s = net_inh / (float)origin_imgh;
    int neww = origin_imgw * s;
    //printf("%x,%d,%d,%d,%d,%d,%d\r\n",inputs,origin_imgw, origin_imgh,net_inw, net_inh,net_outw,net_outh);
    if (neww > net_inw)
    {
        neww = net_inw;
        s = neww / (float)origin_imgw;
        newh = origin_imgh * s;
    }
    nms_scale = 1 / s;

    framecnt+=1;
    std::vector<int> tensor_sizes;
    std::vector<char*> i_inputs;
    for(size_t i = 0; i < input_tensor_num; i++) {
        long ptr = (long)inputs[i];
        i_inputs.push_back((char*)ptr);
    }

#if DUMP_OUTDATA
    char filename[50];
    sprintf(filename, "outdata/py_int_%d.txt", framecnt);
    FILE *file_handle = fopen(filename, "w");
#endif
    //get output data from python
    for(int i=0;i< 1*net_outh*57*net_outw;i++) {
#if DUMP_OUTDATA
        fprintf(file_handle, "%d\n", i_inputs[0][i]);
#endif
        netoutdata[i] = i_inputs[0][i]*scale_to_float;
    }

#if DUMP_OUTDATA
    fclose(file_handle);
#endif
    //post process
    float* current;
    std::vector<cv::Mat> ret_images;
    std::vector<float> prediction;
    for (int i=0;i< input_tensor_num;i++) {
        current = netoutdata + i*net_outh*57*net_outw;
        cv::Mat netimage=cv::Mat(net_outh*57,net_outw ,CV_32FC1,current).clone();
        //cv::imwrite("feature.jpg", netimage);
        ret_images.push_back(netimage);
        prediction = NMSCON(netimage, nms_scale, net_inh, net_inw);
    }
    //if (!prediction.empty())
        //std::cout << "prediction numbers " << prediction.size() << std::endl;

#if !SHOW_FROME_FILE
    std::vector<cv::Mat> channels;
    int offset = 0;
    offset = origin_imgh * origin_imgw;
    cv::Mat channel0(origin_imgh, origin_imgw, CV_8UC1, (void *)(ori_image + offset*0));
    channels.push_back(channel0);
    cv::Mat channel1(origin_imgh, origin_imgw, CV_8UC1, (void *)(ori_image + offset*1));
    channels.push_back(channel1);
    cv::Mat channel2(origin_imgh, origin_imgw, CV_8UC1, (void *)(ori_image + offset*2));
    channels.push_back(channel2);
    cv::Mat origin(origin_imgh, origin_imgw, CV_8UC3);
    cv::merge(channels, origin);
    //cv::imwrite("python.jpg", origin);
    paint_image_single(origin, prediction);
#else
    cv::Mat origin;
    paint_image_single(origin, prediction);
#endif

    return prediction.size();
}


void paint_image_single(cv::Mat &ori_img, std::vector<float> &predictions)
{
    int net_n = 1;
    cv::Mat paintImgs;
    int window_xs = 100;
    int window_ys = 100;
    int window_ws = 800;
    int window_hs = 500;
#if !SHOW_FROME_FILE
    //std::cout << "origin " << ori_img.rows << " " << ori_img.cols << std::endl;
    if (predictions.empty()){
       paintImgs = ori_img.clone();
    }else {
       paintImgs = (PaintImage(ori_img, predictions)).clone();
    }
#else
    cv::Mat readimg = cv::imread("tensor.jpg", 1);
    if (readimg.empty()) {
        std::cout << "failed to read image !" << std::endl;
    }
    if (predictions.empty()){
       paintImgs = readimg.clone();
       //cv::imshow("show", paintImgs);
    }else {
       paintImgs = (PaintImage(readimg, predictions)).clone();
    }
#endif
    //std::cout << "after paint" << std::endl;
    for (int i=0;i<net_n;i++){
      std::string window_name = "Pose_" + std::to_string(i);
      cv::namedWindow(window_name, CV_WINDOW_NORMAL);
      cv::moveWindow(window_name, window_xs, window_ys);
      cv::resizeWindow(window_name, window_ws, window_hs);
      cv::imshow(window_name, paintImgs);
      char c = cvWaitKey(1);
      if(c==27) std::terminate();
      //cv::imshow("output", paintImgs);
      //cv::imwrite("./output.jpg", paintImgs);
    }
}

extern "C" {
int post_process(
        long inputs[],
        int net_n,
        int origin_imgw,
        int origin_imgh,
        int net_inw,
        int net_inh,
        int net_outw,
        int net_outh,
        int input_tensor_num,
        int out_size,
        unsigned char* ori_image)
{
   return post_process_1(
        inputs,
        net_n,
        origin_imgw,
        origin_imgh,
        net_inw,
        net_inh,
        net_outw,
        net_outh,
        input_tensor_num,
        out_size,
        ori_image);
}
void post_process_hello(void)
{    printf("## load post process ##\r\n");
}
}

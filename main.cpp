#include <stdio.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"


static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];

void extract_features_from_frame(const cv::Mat& frame) {
    cv::Mat processed;
    cv::resize(frame, processed, cv::Size(EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT));
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    if (processed.isContinuous()) {
        const uint8_t* p = processed.ptr<uint8_t>();
        for (int i = 0; i < processed.total(); ++i) {
            //rgb to uint32_t
            uint32_t red = *p++;
            uint32_t green = *p++;
            uint32_t blue = *p++;
            features[i] = (red << 16) | (green << 8) | blue;
        }
    }
}

// Callback function to provide data to the model
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) return -1; 
    for (size_t i = 0; i < length; ++i) {
        out_ptr[i] = static_cast<float>(features[offset + i]);
    }
    return EIDSP_OK;
}


void runInference() {
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) { 
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    /* FOR WRITING VIDEO
    cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(width, height));
    if (!video.isOpened()) {
        std::cerr << "ERROR: Could not open video writer" << std::endl;
        return;
    }
    */

    cv::namedWindow("Live Detection", cv::WINDOW_AUTOSIZE);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    cv::Mat frame;
    signal_t signal;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;

    while (true) {
        cap >> frame; 
        if (frame.empty()) break;

        float xScale = frame.cols / EI_CLASSIFIER_INPUT_WIDTH;
        float yScale = frame.rows / EI_CLASSIFIER_INPUT_HEIGHT;

        extract_features_from_frame(frame);
        signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data = &get_signal_data;

        res = run_classifier(&signal, &result, false);
        if (res != EI_IMPULSE_OK) {
            std::cerr << "ERROR: Failed to run classifier" << std::endl;
            break; 
        }

        for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value == 0) {
                continue;
            }

            int x = (bb.x + bb.width / 2) * xScale;
            int y = (bb.y + bb.height / 2) * yScale;
            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }
        
        cv::imshow("Live Detection", frame);
        //video.write(frame); 

        if (cv::waitKey(1) == 27) break; // ESC key
    }

    cap.release(); 
    //video.release();
    cv::destroyAllWindows();
}


int main() {
    runInference();
    return 0;
}

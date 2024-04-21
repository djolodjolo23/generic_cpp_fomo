#include <stdio.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"


static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) return -1; // Handle buffer overflow
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = features[offset + i];
    }
    return EIDSP_OK;
}



void captureAndProcess() {
    cv::VideoCapture cap(0); // open cam
    if (!cap.isOpened()) { 
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return;
    }

    cv::namedWindow("Live Detection", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    signal_t signal;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;


    while (true) {
        cap >> frame; 
        if (frame.empty()) break;

        // Calculate scale factors
        float xScale = frame.cols / 160.0;
        float yScale = frame.rows / 160.0;

        cv::Mat processed;
        cv::Rect roi((frame.cols - frame.rows) / 2, 0, frame.rows, frame.rows); // Crop to square from the center
        cv::Mat cropped = frame(roi);

        // Resize the cropped image
        cv::resize(cropped, processed, cv::Size(160, 160));


        // Convert Mat to float array (features)
        if (processed.isContinuous()) {
            std::memcpy(features, processed.data, processed.total() * processed.elemSize());
        } else {
            // If the matrix is not continuous, we need to copy it row by row
            auto *p_features = features;
            for (int i = 0; i < processed.rows; ++i) {
                std::memcpy(p_features, processed.ptr<float>(i), processed.cols * processed.elemSize());
                p_features += processed.cols * processed.elemSize() / sizeof(float);
            }
        }
 
        signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data = &get_signal_data;

        res = run_classifier(&signal, &result, false);  


        if (res != EI_IMPULSE_OK) {
            std::cerr << "ERROR: Failed to run classifier" << std::endl;
            return;
        }


        for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value < 0.85) {
                continue;
            }

            int x = bb.x + bb.width / 2;
            int y = bb.y + bb.height / 2;

            ei_printf("unscaled x: %d, y: %d\r\n", x, y);

            // Scale the coordinates back to the original frame
            x = x * xScale;
            y = y * yScale;

            ei_printf("scaled x: %d, y: %d\r\n", x, y);
            
            ei_printf("Object detected!:\r\n");
            // print coordinates
            ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                    bb.label,
                    bb.value,
                    bb.x,
                    bb.y,
                    bb.width,
                    bb.height);


            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
            ei_printf("Drawing circle at x: %d, y: %d\r\n", x, y);
            cv::putText(frame, bb.label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
        

        cv::imshow("Live Detection", frame);

        if (cv::waitKey(1) == 27) break; // Break the loop on pressing 'ESC' 

    }

    cap.release(); // When everything done, release the video capture object
    cv::destroyAllWindows(); // Close all OpenCV windows
}


int main() {
    captureAndProcess();
    return 0;
}

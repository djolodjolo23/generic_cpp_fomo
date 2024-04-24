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
    if (offset + length > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) return -1; // Handle buffer overflow
    for (size_t i = 0; i < length; ++i) {
        out_ptr[i] = static_cast<float>(features[offset + i]);
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

    ei_printf("Starting inferencing in 2 seconds...\r\n");
    ei_printf("Please point the camera to the object you want to detect\r\n");
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    ei_printf("Camera resolution: %d x %d\r\n", width, height);
    std::this_thread::sleep_for(std::chrono::seconds(2));


    cv::Mat frame;
    signal_t signal;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;


    while (true) {
        cap >> frame; 
        if (frame.empty()) break;

        // Calculate scale factors
        float xScale = frame.cols / EI_CLASSIFIER_INPUT_WIDTH;
        float yScale = frame.rows / EI_CLASSIFIER_INPUT_HEIGHT;

        extract_features_from_frame(frame);
        signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data = &get_signal_data;

        res = run_classifier(&signal, &result, false);  


        if (res != EI_IMPULSE_OK) {
            std::cerr << "ERROR: Failed to run classifier" << std::endl;
            return;
        }

        for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value == 0) {
                continue;
            }

            ei_printf("count of bounding boxes: %d\r\n", result.bounding_boxes_count);

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
            //ei_printf("Drawing circle at x: %d, y: %d\r\n", x, y);
            //cv::putText(frame, bb.label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
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

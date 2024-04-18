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

        cv::Mat processed;
        cv::resize(frame, processed, cv::Size(160, 160)); 

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
        //std::cout << "features size: " << sizeof(features) << std::endl; // 409600, seems like this is correct since in c++ returns total number of bytes occupied bty the array in mem
 
        signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data = &get_signal_data;

        res = run_classifier(&signal, &result, false);

        float xScale = (float)frame.cols / 160;
        float yScale = (float)frame.rows / 160;

        if (res != EI_IMPULSE_OK) {
            std::cerr << "ERROR: Failed to run classifier" << std::endl;
            return;
        }

        std::cout << "Predictions:\n";

        ei_printf("Object detection bounding boxes:\r\n");
        for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value == 0) {
                continue;
            }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
        }

        // for each bounding box calculate centroid coordinates 
        // and draw a circle on the frame

        uint32_t xy[EI_CLASSIFIER_OBJECT_DETECTION_COUNT][2]; // [x, y]

        uint32_t num_objects = 0;
        uint32_t max_x = EI_CLASSIFIER_OBJECT_DETECTION_COUNT;

        for (size_t ix = 0; ix < max_x; ix++) {
            auto bb = result.bounding_boxes[ix];
            if (bb.value == 0) {
                continue;
            }
            xy[num_objects][0] = bb.x + bb.width / 2;
            xy[num_objects][1] = bb.y + bb.height / 2;
            num_objects++;
        }

        for (uint32_t i = 0; i < num_objects; i++) {
            int x = xy[i][0];
            int y = xy[i][1];

            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }

        /*
        for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value == 0) {
                continue;
            }
            int x = bb.x + bb.width / 2 * xScale;
            int y = bb.y + bb.height / 2 * yScale;

            cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }
        */

        cv::imshow("Live Detection", frame);

        // Wait for a key press for 1 millisecond to see if user wants to exit
        if (cv::waitKey(1) == 27) break; // Break the loop on pressing 'ESC' 

    }

    cap.release(); // When everything done, release the video capture object
    cv::destroyAllWindows(); // Close all OpenCV windows
}


int main() {
    captureAndProcess();
    return 0;
}

/*

int main(int argc, char **argv) {

    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

    // Calculate the length of the buffer
    size_t buf_len = sizeof(features) / sizeof(features[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("ERROR: The size of the input buffer is not correct.\r\n");
        ei_printf("Expected %d items, but got %d\r\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
                (int)buf_len);
        return 1;
    }

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;

    // Perform DSP pre-processing and inference
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    ei_printf("run_classifier returned: %d\r\n", res);
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
#endif

    return 0;
}
*/

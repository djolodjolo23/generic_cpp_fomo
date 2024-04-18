#include <stdio.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"


static const int FEATURE_SIZE = 102400; // feature size for the model
static float features[FEATURE_SIZE];

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > FEATURE_SIZE) return -1; // Handle buffer overflow

    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = features[offset + i];
    }
    return EIDSP_OK;
}

void captureAndProcess() {
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {   // Check if we succeeded
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return;
    }

    cv::Mat frame;
    signal_t signal;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;


    while (true) {
        cap >> frame; // Capture a new frame
        if (frame.empty()) break;

        cv::Mat processed;
        cv::cvtColor(frame, processed, cv::COLOR_BGR2GRAY);
        cv::resize(processed, processed, cv::Size(FEATURE_SIZE, 1)); 

        // Convert Mat to float array (features)
        for (int i = 0; i < processed.total(); i++) {
            features[i] = processed.at<unsigned char>(i);
        }

        signal.total_length = FEATURE_SIZE;
        signal.get_data = &get_signal_data;

        res = run_classifier(&signal, &result, false);

        if (res != EI_IMPULSE_OK) {
            std::cerr << "ERROR: Failed to run classifier" << std::endl;
            return;
        }

        std::cout << "Predictions:\n";

        for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            printf("  %s: ", ei_classifier_inferencing_categories[i]);
            printf("%.5f\r\n", result.classification[i].value);
        }

        std::this_thread::sleep_for(std::chrono::seconds(5)); // Sleep for 5 seconds


    }

    cap.release(); // When everything done, release the video capture object
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
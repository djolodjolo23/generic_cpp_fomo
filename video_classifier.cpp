#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thread>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

static float features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE];

void extract_features_from_frame(const cv::Mat& frame) {
    cv::Mat processed;
    // Adjust the target size to match model requirements
    cv::resize(frame, processed, cv::Size(EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT));
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    if (processed.isContinuous()) {
        const uint8_t* p = processed.ptr<uint8_t>();
        for (int i = 0; i < processed.total(); ++i) {
            uint32_t red = *p++;
            uint32_t green = *p++;
            uint32_t blue = *p++;
            features[i] = (red << 16) | (green << 8) | blue;  // This might be incorrect if the model expects normalized inputs
        }
    }
}


static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) return -1; 
    for (size_t i = 0; i < length; ++i) {
        out_ptr[i] = static_cast<float>(features[offset + i]);
    }
    return EIDSP_OK;
}

// Draw a rectangle and accompanying text on the frame
void drawRectangle(cv::Mat& frame, const ei_impulse_result_bounding_box_t& bb, float xScale, float yScale, const std::string& label, const std::string& value, int font, double fontScale, int thickness) {
    int x = bb.x * xScale;
    int y = bb.y * yScale;
    int w = bb.width * xScale;
    int h = bb.height * yScale;
    cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);
    cv::Size labelSize = cv::getTextSize(label + " " + value, font, fontScale, thickness, nullptr);
    cv::Point labelOrg(x - labelSize.width / 2, y - 10);
    cv::putText(frame, label + " " + value, labelOrg, font, fontScale, cv::Scalar(0, 0, 255), thickness);
}

// Draw a circle and accompanying text on the frame
void drawCircle(cv::Mat& frame, const ei_impulse_result_bounding_box_t& bb, float xScale, float yScale, const std::string& label, const std::string& value, int font, double fontScale, int thickness) {
    int x = (bb.x + bb.width / 2) * xScale;
    int y = (bb.y + bb.height / 2) * yScale;
    cv::circle(frame, cv::Point(x, y), 8, cv::Scalar(0, 0, 255), 2); // Outer circle in red
    cv::circle(frame, cv::Point(x, y), 4, cv::Scalar(0, 0, 255), -1); // Inner circle in red
    cv::Size textSize = cv::getTextSize(label + " " + value, font, fontScale, thickness, nullptr);
    cv::Point textOrg(x - textSize.width / 2, y - 10); // Adjust position to place label on top
    cv::putText(frame, label + " " + value, textOrg, font, fontScale, cv::Scalar(0, 0, 255), thickness);
}

void runInference(const std::string& videoPath) {
    cv::VideoCapture cap(videoPath); 
    if (!cap.isOpened()) { 
        std::cerr << "ERROR: Could not open video file: " << videoPath << std::endl;
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    
    cv::VideoWriter video("output/yolo_unseen_kalmar_13.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(width, height));
    if (!video.isOpened()) {
        std::cerr << "ERROR: Could not open video writer" << std::endl;
        return;
    }
    

    cv::namedWindow("Video Detection", cv::WINDOW_AUTOSIZE);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    cv::Mat frame;
    signal_t signal;
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;

    while (true) {
        cap >> frame; 
        if (frame.empty()) break;

        float xScale = frame.cols / 320.0f;
        float yScale = frame.rows / 320.0f;


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
            if (bb.value < 0.5f) {
                continue;
            }

            std::string label = "queen";
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << bb.value;
            std::string value = ss.str();

            int font = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;

            drawRectangle(frame, bb, xScale, yScale, label, value, font, fontScale, thickness);
            // Uncomment the next line if you need to draw circles as well
            //drawCircle(frame, bb, xScale, yScale, label, value, font, fontScale, thickness);
        }

        cv::imshow("Video Detection", frame);
        video.write(frame); 


        if (cv::waitKey(1) == 27) break; // ESC key
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video file path>" << std::endl;
        return -1;
    }
    std::string videoPath = argv[1];
    runInference(videoPath);
    return 0;
}
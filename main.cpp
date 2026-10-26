#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

using namespace cv;
using namespace cv::dnn;
namespace fs = std::filesystem;

#define DETECT_INTERVAL 10 // Re-run detect after each 10 frames
#define IOU_THRESHOLD 0.4  // Ngưỡng IoU để gán detection cho tracker
#define EMO_W 64           // Input width of MobileNet
#define EMO_H 64           // Input height of MobileNet
const std::string LOG_ROOT = "face_logs"; // Result folder

double calculateIoU(const Rect2d& rect1, const Rect2d& rect2) {
    Rect intersection = rect1 & rect2;
    double area_i = intersection.area();
    double area_u = rect1.area() + rect2.area() - area_i;
    if (area_u > 0) return area_i / area_u;
    return 0.0;
}

int main(int argc, char** argv) {
    const cv::String keys =
    "{help h usage| | print this message}"
    "{input i |0| path to input video file or webcam ID}";
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) { parser.printMessage(); return 0; }
    std::string input = parser.get<cv::String>("input");

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // 0. Create result folder if it does not exist
    if (!fs::exists(LOG_ROOT)) {
        if (fs::create_directory(LOG_ROOT))
            std::cout << "Created folder " << LOG_ROOT << ".\n";
        else return -1;
    }
    // Create new folder save the result of i-th video
    int i = 1;
    for (const auto& entry : fs::directory_iterator(LOG_ROOT)) i++;
    std::string LOG_DIR = LOG_ROOT + "/" + std::to_string(i);
    if (fs::create_directory(LOG_DIR))
        std::cout << "Created folder " << i << ".\n";
    else return -1;


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // 1. YuNet configuration: for detect face
    const int inputW = 640;
    const int inputH = 480;
    std::string yunetPath = std::string(PROJECT_ROOT) + "/models/face_detection_yunet_2023mar_int8.onnx";
    Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
        yunetPath, "",
        Size(inputW, inputH), 0.78f, 0.3f, 50
        // 0.78f: confidence threshold: only take prediction that has confidence score > confidence threshold
        // 0.3f: IoU threshold for Non-Maximum Suppression
        // 50: only take maximum 50 predictions
    );
    detector->setInputSize(Size(inputW, inputH));
    if (!detector) {
        std::cerr << "Can't create FaceDetectorYN. Check ONNX model directory." << std::endl;
        return -1;
    }

    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // 2. MobileNet configuration: for classify emotion
    std::string mobilenetPath = std::string(PROJECT_ROOT) + "/models/MobileNet.onnx";
    Net emotionNet = readNetFromONNX(mobilenetPath);
    if (emotionNet.empty()) {
        std::cerr << "Can't load emotion classification model." << std::endl;
        return -1;
    }
    emotionNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    emotionNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::vector<std::string> emotion_labels = { "Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise" };
    int num_classes = emotion_labels.size();


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // 3. Open camera, configurate parameters for tracking
    VideoCapture cap; // 0: webcam. Can pass a directory to a video, like "video.mp4"
    if (input.size() == 1 && isdigit(input[0]))
        cap.open(stoi(input));
    else 
        cap.open(input);
    if (!cap.isOpened()) {
        std::cerr << "Can't open webcam or file video!" << std::endl;
        return -1;
    }
    Mat frame, img;
    int frame_count = 0;
    long next_face_id = 0;
    std::map<long, std::pair<Ptr<Tracker>, Rect2d>> active_trackers;

    std::map<long, std::ofstream> emotion_logs; // Map for retaining all CSV files are opened
    // Vector 2D for saving average emotion of each person (each ID), format: [[Angry, ..., Surprise, Num_frame],...]
    std::vector<std::vector<float>> average;
    std::cout << "Start YuNet Detection and Tracking. Press ESC to exit." << std::endl;
    Mat face;


    // - - - - - Compare with result on previous frame to avoid noise - - - - - //
    Mat pre_prob = Mat::zeros(1, 8, CV_32F);
    Mat diff, prob;
    double distSq;
    float threshold = 0.3;
    // - - - - - //


    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        // Resize frame to match YuNet's input
        resize(frame, frame, Size(inputW, inputH), 0, 0, INTER_AREA);


        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        // 4. Detect face use YuNet: for each DETECT_INTERVAL frame or when no face are in tracking
        //    and update trackers
        if (frame_count % DETECT_INTERVAL == 0 || active_trackers.empty()) {
            Mat faces;
            detector->detect(frame, faces);
            std::vector<Rect2d> current_detections;
            if (!faces.empty()) {
                for (int i = 0; i < faces.rows; i++) {
                    float x = faces.at<float>(i, 0);
                    float y = faces.at<float>(i, 1);
                    float w = faces.at<float>(i, 2);
                    float h = faces.at<float>(i, 3);
                    current_detections.emplace_back(x, y, w, h);
                }
            }

            // A. Remove trackers that are lose track (do not match with any new prediction): based on IoU threshold
            std::vector<long> trackers_to_remove;
            for (auto const& pair_item : active_trackers) {
                long id = pair_item.first;
                const auto& tracker_data = pair_item.second;
                bool found_overlap = false;
                for (const auto& det_rect : current_detections) {
                    if (calculateIoU(tracker_data.second, det_rect) > IOU_THRESHOLD) {
                        found_overlap = true;
                        break;
                    }
                }
                if (!found_overlap) {
                    trackers_to_remove.push_back(id);
                }
            }
            for (long id : trackers_to_remove) {
                // Close file when tracker is removed
                if (emotion_logs.count(id)) {
                    emotion_logs[id].close();
                    emotion_logs.erase(id);
                }
                active_trackers.erase(id);
            }

            // B. Update and add new trackers
            for (const auto& new_face_box : current_detections) {
                bool found_existing_tracker = false;
                for (auto& pair_item : active_trackers) {
                    auto& tracker_data = pair_item.second;
                    // Update bbox of tracker that match with prediction
                    if (calculateIoU(tracker_data.second, new_face_box) > IOU_THRESHOLD) {
                        tracker_data.second = new_face_box;
                        tracker_data.first->init(frame, tracker_data.second);
                        found_existing_tracker = true;
                        break;
                    }
                }

                // If no tracker match with prediction we are considering -> It's a new face - add new tracker for it
                if (!found_existing_tracker) {
                    Ptr<Tracker> new_tracker = TrackerNano::create();
                    if (new_tracker) {
                        new_tracker->init(frame, new_face_box);

                        // Create result folder for new tracker
                        long new_id = next_face_id;
                        std::string id_str = "ID" + std::to_string(new_id);
                        std::string face_dir_path = LOG_DIR + "/" + id_str;
                        if (fs::create_directories(face_dir_path)) {
                            std::cout << "Created folder for " << id_str << std::endl;
                        }

                        // Save first frame with bbox
                        Mat first_frame_copy = frame.clone();
                        rectangle(first_frame_copy, new_face_box, Scalar(0, 0, 255), 2);
                        cv::putText(first_frame_copy, id_str, Point(new_face_box.x, new_face_box.y - 10),
                            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                        imwrite(face_dir_path + "/first_frame.jpg", first_frame_copy);

                        // Create and open CSV file
                        std::ofstream csv_file(face_dir_path + "/emotions.csv");
                        if (csv_file.is_open()) {
                            csv_file << "frame,Angry,Contempt,Disgust,Fear,Happy,Neutral,Sad,Surprise\n"; // Write header
                            emotion_logs[new_id] = std::move(csv_file); // Add file id to map
                        }
                        else std::cerr << "Can't create file CSV for " << id_str << std::endl;
                        active_trackers[next_face_id++] = { new_tracker, new_face_box };
                    }
                    else std::cerr << "Can't create new Tracker Nano!" << std::endl;
                }
            }
        }


        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        // 5. Tracking
        else {
            std::vector<long> failed_trackers;
            for (auto& pair_item : active_trackers) {
                long id = pair_item.first;
                auto& tracker_data = pair_item.second;
                Rect current_roi = tracker_data.second;
                bool ok = tracker_data.first->update(frame, current_roi);
                if (ok) tracker_data.second = current_roi;
                else failed_trackers.push_back(id);
            }
            for (long id : failed_trackers) {
                // Close file when tracker is removed
                if (emotion_logs.count(id)) {
                    emotion_logs[id].close();
                    emotion_logs.erase(id);
                }
                active_trackers.erase(id);
            }
        }


        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        // 6. Classify emotion use MobileNet
        for (auto& pair_item : active_trackers) {
            long id = pair_item.first;
            auto& tracker_data = pair_item.second;
            Rect current_roi = tracker_data.second;

            // Ensure current_roi is valid
            current_roi.x = (std::max)(0, current_roi.x);
            current_roi.y = (std::max)(0, current_roi.y);
            current_roi.width = (std::min)(current_roi.width, frame.cols - current_roi.x);
            current_roi.height = (std::min)(current_roi.height, frame.rows - current_roi.y);
            if (current_roi.width <= 0 || current_roi.height <= 0) continue;

            // Pre-process: convert BGR to RGB with R=G=B, resize and normalize
            face = frame(current_roi);
            cvtColor(face, face, COLOR_BGR2GRAY); // cmt when using pca_svm_fer.onnx
            cvtColor(face, face, COLOR_GRAY2RGB);
            face = dnn::blobFromImage(face, 1.0 / 255.0, Size(EMO_W, EMO_H));

            // Forward and extract output infomation
            emotionNet.setInput(face);
            Mat cur_prob = emotionNet.forward();


            // - - - - - Compare with result on previous frame to avoid noise - - - - - //
            cv::absdiff(cur_prob, pre_prob, diff);
            distSq = cv::norm(diff, cv::NORM_L2SQR);
            if (distSq > threshold) {
                prob = cur_prob;
                cur_prob.copyTo(pre_prob);
            }
            else prob = pre_prob.clone();
            // - - - - - //


            Point classIdPoint;
            double confidence;
            cv::minMaxLoc(prob, nullptr, &confidence, nullptr, &classIdPoint);
            int label_id = classIdPoint.x;

            // Write frame id and emotion to CSV file
            emotion_logs.at(id) << frame_count;
            for (int i = 0; i < prob.cols; i++) {
                float val = prob.at<float>(0, i);
                emotion_logs.at(id) << "," << std::fixed << std::setprecision(2) << val;
            }
            emotion_logs.at(id) << "\n";

            if (average.size() < id + 1) { // Create new result vector for current ID
                average.push_back(prob);
                average[id].push_back(1);
            }
            else { // Update average emotion (result vector)
                for (int i = 0; i < num_classes; i++) {
                    average[id][i] = (average[id][i] * average[id][num_classes] + prob.at<float>(0, i)) / (average[id][num_classes] + 1);
                }
                average[id][num_classes] += 1;
            }

            // Show result (label, confidence) on video
            rectangle(frame, current_roi, Scalar(0, 255, 0), 2, 1);
            cv::putText(frame, "ID: " + std::to_string(id), Point(current_roi.x, current_roi.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
            std::string label = emotion_labels[label_id] + format(" (%.2f)", confidence);
            cv::putText(frame, label, Point(current_roi.x, current_roi.y + 5),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        }

        cv::putText(frame, "Frame: " + std::to_string(frame_count) + " | Trackers: " + std::to_string(active_trackers.size()),
            Point(10, frame.rows - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        cv::imshow("YuNet Detection + Tracking Nano + MobileNet classification", frame);
        if (waitKey(1) == 27) break; // Print Esc to exit
        frame_count++;
    }
    // Close all CSV files
    for (auto& pair_item : emotion_logs) {
        if (pair_item.second.is_open()) {
            pair_item.second.close();
        }
    }
    std::cout << "Closed all log file." << std::endl;

    // Show average emotion of each ID on screen
    for (int i = 0; i < average.size(); i++) {
        for (int j = 0; j < average[i].size(); j++) {
            std::cout << average[i][j] << " ";
        }
        std::cout << "\n";
    }
    cap.release();
    cv::destroyAllWindows();

    // Write average emotion to CSV file
    std::string filename = LOG_DIR + "/" + "output.csv";
    std::ofstream csv_file(filename);
    if (csv_file.is_open()) csv_file << "ID,Angry,Contempt,Disgust,Fear,Happy,Neutral,Sad,Surprise,Num_frames\n"; // Header
    else std::cerr << "Can't create file CSV for " << filename << "\n";

    int id_temp = 0;
    for (const auto& row : average) {
        csv_file << id_temp << ",";
        for (int i = 0; i < row.size(); i++) {
            csv_file << row[i];
            if (i != row.size() - 1)
                csv_file << ",";
        }
        csv_file << "\n";
        id_temp++;
    }

    csv_file.close();
    std::cout << "Writed data to " << filename << " completed." << std::endl;
    return 0;
}
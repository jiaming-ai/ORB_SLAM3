/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<sstream>
#include<string>
#include<dirent.h>

#include<opencv2/core/core.hpp>
#include<opencv2/videoio.hpp>

#include<System.h>

using namespace std;

void LoadFrameIds(const string &strFile, vector<int> &vFrameIds,
                 vector<double> &vTimestamps);
string FindTimestampFile(const string &directory);
string FindVideoFile(const string &directory);

int main(int argc, char **argv)
{
    bool useDepth = false;
    string sequencePath;
    
    // Check for -d flag
    if(argc == 5 && string(argv[4]) == "-d")
    {
        useDepth = true;
        sequencePath = string(argv[3]);
    }
    else if(argc == 4)
    {
        sequencePath = string(argv[3]);
    }
    else
    {
        cerr << endl << "Usage: ./mono_frodo path_to_vocabulary path_to_settings path_to_sequence [-d]" << endl;
        cerr << "  -d: Use depth images from path_to_sequence/depth/" << endl;
        return 1;
    }

    // Find timestamp file with pattern front_camera_timestamps_xxx.csv
    string timestampFile = FindTimestampFile(sequencePath);
    if (timestampFile.empty()) {
        cerr << "Could not find timestamp file in directory: " << sequencePath << endl;
        return 1;
    }
    cout << "Using timestamp file: " << timestampFile << endl;

    // Retrieve frame IDs and timestamps
    vector<int> vFrameIds;
    vector<double> vTimestamps;
    LoadFrameIds(timestampFile, vFrameIds, vTimestamps);

    int nFrames = vFrameIds.size();
    if (nFrames == 0) {
        cerr << "No frames found in timestamp file" << endl;
        return 1;
    }

    // Find video file with pattern xxx_ride_xxx__uid_s_1000__uid_e_video.m3u8
    string videoPath = FindVideoFile(sequencePath + "/recordings");
    if (videoPath.empty()) {
        cerr << "Could not find video file in directory: " << sequencePath + "/recordings" << endl;
        return 1;
    }
    cout << "Using video file: " << videoPath << endl;

    // Open video stream
    cv::VideoCapture videoCapture(videoPath);
    
    if (!videoCapture.isOpened()) {
        cerr << "Failed to open video stream at: " << videoPath << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], 
                          useDepth ? ORB_SLAM3::System::RGBD : ORB_SLAM3::System::MONOCULAR,
                          true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nFrames);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Frames in the sequence: " << nFrames << endl;
    if (useDepth) {
        cout << "Using RGBD mode with depth images from: " << sequencePath << "/depth/" << endl;
    } else {
        cout << "Using Monocular mode" << endl;
    }
    cout << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im, imD;
    int lastFrameId = -1;
    
    for(int ni=0; ni<nFrames; ni++)
    {
        int currentFrameId = vFrameIds[ni];
        double tframe = vTimestamps[ni];
        
        // Check if frames are sequential as expected
        if (currentFrameId == lastFrameId + 1) {
            // Sequential frame, just read the next frame
            videoCapture.read(im);
            lastFrameId = currentFrameId;
        } else {
            // Non-sequential frame, need to seek
            if (currentFrameId < lastFrameId) {
                // If we need to go back, reset and seek forward
                videoCapture.release();
                videoCapture.open(videoPath);
                if (!videoCapture.isOpened()) {
                    cerr << "Failed to reopen video stream" << endl;
                    return 1;
                }
                lastFrameId = -1;
            }
            
            // Skip frames until we reach the desired frame
            while (lastFrameId < currentFrameId - 1) {
                videoCapture.grab();
                lastFrameId++;
            }
            
            // Read the frame
            videoCapture.read(im);
            lastFrameId = currentFrameId;
        }

        if(im.empty())
        {
            cerr << endl << "Failed to load frame " << currentFrameId 
                 << " from video stream: " << videoPath << endl;
            return 1;
        }

        // Load depth image if using depth mode
        if(useDepth)
        {
            string depthImagePath = sequencePath + "/depth/" + to_string(currentFrameId) + ".png";
            imD = cv::imread(depthImagePath, cv::IMREAD_UNCHANGED);
            
            if(imD.empty())
            {
                cerr << endl << "Failed to load depth image at: " << depthImagePath << endl;
                return 1;
            }
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            
            if(useDepth)
                cv::resize(imD, imD, cv::Size(width, height));
                
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        if(useDepth)
            SLAM.TrackRGBD(im, imD, tframe);
        else
            SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nFrames-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nFrames; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nFrames/2] << endl;
    cout << "mean tracking time: " << totaltime/nFrames << endl;

    // Save camera trajectory
//     SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryEuRoC(string(argv[3]) + "/TrajFromBiggestMap.txt");
//     SLAM.SaveTrajectoryTUM("Trajectory.txt");

    return 0;
}

void LoadFrameIds(const string &strFile, vector<int> &vFrameIds, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());
    
    if(!f.is_open()) {
        cerr << "Failed to open timestamp file: " << strFile << endl;
        return;
    }

    // Skip header line
    string header;
    getline(f, header);
    
    // Check if header matches expected format
    if (header.find("frame_id") == string::npos || header.find("timestamp") == string::npos) {
        cout << "Warning: CSV header doesn't match expected format. Expected: frame_id,timestamp" << endl;
    }
    
    string line;
    while(getline(f, line)) {
        if(!line.empty()) {
            stringstream ss(line);
            string token;
            
            // Parse frame ID (first column)
            getline(ss, token, ',');
            int frameId = stoi(token);
            vFrameIds.push_back(frameId);
            
            // Parse timestamp (second column)
            getline(ss, token, ',');
            double timestamp = stod(token);
            vTimestamps.push_back(timestamp);
        }
    }
    
    f.close();
}

string FindTimestampFile(const string &directory)
{
    string pattern = "front_camera_timestamps_";
    string result;
    
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;
            if (filename.find(pattern) != string::npos && filename.find(".csv") != string::npos) {
                result = directory + "/" + filename;
                break;
            }
        }
        closedir(dir);
    }
    
    return result;
}

string FindVideoFile(const string &directory)
{
    string pattern = "_ride_";
    string suffix = "__uid_s_1000__uid_e_video.m3u8";
    string result;
    
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;
            if (filename.find(pattern) != string::npos && filename.find(suffix) != string::npos) {
                result = directory + "/" + filename;
                break;
            }
        }
        closedir(dir);
    }
    
    return result;
}

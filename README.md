# Emotion Analysis from Video

Detect each face's emotion from a video/webcam, save the emotional and temporal information for analysis. Eight detected emotions: Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Prerequisites

### Windows
#### 1. Visual Studio Build Tools: Install "Desktop development with C++" with the following workloads/components:
- MSVC
- Windows SDK
- C++ CMake tools for Windows
#### 2. Install CMake, OpenCV 4.x, VS Code with CMake Tools extension.
#### 3. Configure Environment Variables:
- Search for "Edit the system environment variables"
- Click Environment Variables -> Path (in User/System variables) -> Edit -> New
- Add the path to your OpenCV bin folder (e.g., C:\opencv\build\x64\vc16\bin).

### Linux (Ubuntu/Debian)
Run the following command to install dependencies:
```
sudo apt update
sudo apt install build-essential cmake git pkg-config libopencv-dev libsm6 libxext6 libxrender-dev v4l-utils
```

## Clone repo:

```
git clone https://github.com/KienHCMUS/Emotion-Analysis-from-Video.git
cd Emotion-Analysis-from-Video
```

## Build and Run

### Windows:
1. Open the project in VS Code.  
2. Congigure CMake:  
- Press Ctrl+Shift+P -> CMake: Select a Kit -> Choose Visual Studio Build Tools.
- Press Ctrl+Shift+P -> CMake: Select Variant -> Choose Release.
3. Wait for the configuration to finish ([cmake] -- Build files have been written to...).  
4. Click the Build button in the status bar (bottom left).  
5. Run the application via Terminal:
```
cd build/Release
./MyApp.exe --input="path/to/video"
```

### Linux:
Build and Run:
```
mkdir build && cd build
cmake ..
make
./MyApp --input="path/to/video"
```

## Result:
The result will be:
- Shown real-time
- Saved in "face_logs" folder in project

![Emotion](test.jpg)
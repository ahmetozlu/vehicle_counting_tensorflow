# VEHICLE DETECTION, TRACKING AND COUNTING
This project focuses on "Vechicle Detection, Tracking and Counting" by [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). This project has more than just counting vehicles, here are the additional capabilities of this project;

- Recognition of approximate vehicle color
- Detection of vehicle's direction of travel
- Prediction the speed of the vehicle
- Prediction of approximate vehicle size
- **The images of detected vehicles are cropped from video frame and they are saved as new images under "[detected_vehicles](https://github.com/ahmetozlu/vehicle_counting_tensorflow/tree/master/detected_vehicles)" folder path**
- **The program gives a .csv file as an output ([traffic_measurement.csv](https://github.com/ahmetozlu/vehicle_counting_tensorflow/blob/master/traffic_measurement.csv)) which includes "Vehicle Type/Size", " Vehicle Color", " Vehicle Movement Direction", " Vehicle Speed (km/h)" rows, after the end of the process for the source video file.**

The input video can be accessible by this [link](https://github.com/ahmetozlu/vehicle_counting_tensorflow/blob/master/sub-1504614469486.mp4).

## Quick Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/36344830-095cc4ec-1431-11e8-8e57-976c40d87cf9.gif">
</p>

## Theory

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/35445395-8dba4406-02c2-11e8-84bf-b480edbe9472.jpg">
</p>

TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

## Project Demo

## Installation

**1.) Python and pip**

Python is automatically installed on Ubuntu. Take a moment to confirm (by issuing a python -V command) that one of the following Python versions is already installed on your system:

- Python 2.7
- Python 3.3+

The pip or pip3 package manager is usually installed on Ubuntu. Take a moment to confirm (by issuing a *pip -V* or *pip3 -V* command) that pip or pip3 is installed. We strongly recommend version 8.1 or higher of pip or pip3. If Version 8.1 or later is not installed, issue the following command, which will either install or upgrade to the latest pip version:

    $ sudo apt-get install python-pip python-dev   # for Python 2.7
    $ sudo apt-get install python3-pip python3-dev # for Python 3.n
    
**2.) OpenCV**

See required commands to install OpenCV on Ubuntu in [here](https://gist.github.com/dynamicguy/3d1fce8dae65e765f7c4).

**3.) TensorFlow**

Install TensorFlow by invoking one of the following commands:

    $ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
    $ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
    $ pip install tensorflow-gpu  # Python 2.7;  GPU support
    $ pip3 install tensorflow-gpu # Python 3.n; GPU support

**4.) TensorFlow Object Detection API**

See required commands to install TensorFlow Object Detection API on Ubuntu in [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{vdtct,
        author = "Ahmet Özlü",
        title  = "Vehicle Detection, Tracking and Counting by TensorFlow",
        year   = "2018",
        url    = "https://github.com/ahmetozlu/vehicle_counting_tensorflow"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.

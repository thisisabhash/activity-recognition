**Activity Phase Detection in Cholecystectomy Surgery Videos [Done as part of CS766 course at UW-Madison]**

For more information and results, please check the project website - [Website](https://sites.google.com/wisc.edu/cs766-activity-phase-detection/introduction?authuser=1)

This codebase is built upon the code present here - https://github.com/JayJayBinks1/DeepPhase

So, I would like to acknowledge and credit the user above for the code.

Instructions to run the code
----------------------------------

1. Download the dataset from http://camma.u-strasbg.fr/datasets
2. Put all videos in a yourpath/cholec80/Data folder, where yourpath is a local path on your machine.
This can be changed in config.py file with DATA_DIR value.

For example, all videos will have path like -
yourpath/cholec80/Data/video01.mp4

Tool annotations will have path like - 
yourpath/cholec80/Data/video01-tool.txt

Phase annotations will have path like -
yourpath/cholec80/Data/video01-phase.txt

Timestamps will have path like - 
yourpath/cholec80/Data/video01-timestamp.txt

3. Run frame_extraction.py to generate frames from videos.
4. Run tool_recognition.py for tool recognition.
5. Run feature_extract.py for feature extraction.
6. Run phase_recognition.py for phase recognition.

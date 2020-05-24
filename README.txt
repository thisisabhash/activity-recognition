README
---------------------

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
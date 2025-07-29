This folder contains the data relevant to the program as follows:
1) Labels 05-17-2024_modified.xlsx -> This is the ground truth behavior label of the cows
2) cow_behavior_front_20240517064400.csv -> The predicted behavior from the front camera using the code IoU_Front_update.py
3) cow_behavior_top_IoU_20240517064400_0.8_0.5 -> The predicted behavior from the top camera using the code IoU_Top_update.py
4) cow-name-label-a02.v1i.yolov8.zip -> Data for training the top camera cow identity detection model
5) cow-label-a04-name.v1i.yolov8.zip -> Data for training the front camera cow identity detection model
6) cow-label-a04-activity-clone.v11i.yolov8.zip -> Data for training the front camera cow behavior detection model
7) cow-label-a03-activity.v2i.yolov8.zip ->  Data for training the top camera cow activity detection model
8) A04_20240517035610.mp4, A04_20240517131349.mp4,  -> video of front camera for testing the model
9) A03_20240517074312.mp4, A03_20240517000906.mp4, A03_20240517185637.mp4 -> video of top camera for testing the model

Due to size limit of Github, dataset 5)-8) can be retrieved from this directory
https://drive.google.com/drive/folders/1VvruYwSxbmAfp7cHAehJ5_zYTIsvHBEy

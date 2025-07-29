This folder contains the code as follows:
1) IoU_Front_update.py -> Calculate the IoU between the cow identity detection model and cow behavior detection model of the front camera
   - Input:
      - front_source: Video source for applying the YOLO model
      - front_act_model: YOLO model for front behavior detection 
      - front_iden_model: YOLO model for front identity detection
   - Output:
      - csv_file: The csv file that contains the mapping between behavior and identity of each timestamp
2) IoU_Top_update.py -> Calculate the IoU between the cow identity detection model and cow behavior detection model of the top camera
   - Input:
      - front_source: Video source for applying the YOLO model
      - front_act_model: YOLO model for top behavior detection 
      - front_iden_model: YOLO model for top identity detection
   - Output:
      - csv_file: The csv file that contains the mapping between behavior and identity of each timestamp 
3) Compare_ensemble_with_single_camera_PLOS.ipynb -> Calculate the accuracy of the model compared with ground truth data, and then calculate F1-score and confusion matrix of the estrus and non-estrus behaviors
   - Input:
      - labels_df: ground truth behavioral data
      - predictions_df_front: front prediction csv file data obtained from IoU_Front_update.py
      - predictions_df_top: front prediction csv file data obtained from IoU_Front_update.py
   - Output results:
      - F1-score 
      - Confusion matrix
4) train_model.py -> code to train the YOLOv8 model
    - Input: 
      - data.yaml: setting file for training the model, which links to the dataset
    - Output:  
      - model

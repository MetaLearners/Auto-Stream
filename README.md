# Auto-Stream
Auto-Stream is an AutoML system designed for streaming data processing. Different from most existing AutoML systems which can only work on single batch learning problems, Auto-Stream can work in a online fashion by automatically evolving over time and adapting to concept drift in data streams. 

## Code Structure
The current version is developed based on the code skeleton of NeurIPS 2018 AutoML Challenge (please refer to the folder ```AutoML3_ingestion_program```). 
The code of Auto-Stream is in the folder ```Auto-Stream```. 
We also provide several baseline methods in the folder ```baselines```. 

## How to Use
Use the following commend the run your test: 
```
python3 AutoML3_ingestion_program/ingestion.py data_folder prediction_folder AutoML3_ingestion_program code_folder
```
where ```data_folder``` is the path to the data, ```prediction_folder``` is the path to save prediction results, and ```code_folder``` is the path to the code of the AutoML system you want to use. 
The dataset should be organized in a specific way to be compitable with the interfaces in the ingestion program. We give a example of how to organize the data in folder ```data```. In general, you should create a ```X_feat.type``` file to store the feature types of each feature column, a ```X_public.info``` to store some meta data about the dataset. The features and labels of each data batch should be stored separately as ```X_test.data``` and ```X_test.solution```. The data of the first batch should be named as ```X_train1.data``` and ```X_train1.solution```, because they are used for the initial training of the AutoML system. 
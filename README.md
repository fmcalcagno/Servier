SERVIER DATA SCIENCE TEST
==========================

    This is the README file of the Servier Data Science Test created by Facundo Calcagno (fmcalcagno@gmail.com).
    In the following lines I'll go throw the installation, usage and execution process of the package, and I'll explain some
    of my choices during the project.

## INSTALLATION

1- Install Nvidia Drivers.

    - Download an install Nvidia Drivers using the instruction in the following website: 
      [Nvidia Drivers instructions](https://www.nvidia.com/Download/index.aspx)

2- Install Docker.

    - Install Docker using the instructions in the following website:
      [Docker installer instructions instructions](https://docs.docker.com/engine/install/ubuntu/)

3- Install Nvidia container toolkit.
    
    - Install the Nvidia Container toolkit to be able to use your GPU from inside a Docker container.
      Please follow the instructions in the following website:
      [Nvidia Container Toolkit instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
 
4- Clone the github Repository in the desired folder
    git clone https://github.com/fmcalcagno/Servier

5- Download the trained models and the training and validations sets from the zip I delivered

6- Copy the training and validation sets into: 
    
    */servier/data/

7- Copy the models into the model's folder:
    
    ./servier/models


4-  Build Docker container from servier folder: 
    
    docker build . -t servier



5- Run the desired process (Train / Predict / Evaluate) for the desired model.

    - Use the instructions in the #Usage Chapter to execute the package.

## Parameters

This is the list of parameters for all 3 functionalities: Train, Evaluate and Predict.
They are not all necessary for the 3 functions.

    --modelnumber MODELNUMBER
                        Chose model tu use
    --n_epoch N_EPOCH, -e N_EPOCH
                        number of epochs
    --train_data TRAIN_DATA
                        train corpus (.csv)
    --val_data VAL_DATA   validation corpus (.csv)
    --out_dir_models OUT_DIR_MODELS, -o OUT_DIR_MODELS
                        output directory
    --out_file_results OUT_FILE_RESULTS, -of OUT_FILE_RESULTS
                        output file for Evaluation
    --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size
    --lr LR               Learning rate
    --modelpath MODELPATH
                        Model to load
    --out_model_name OUT_MODEL_NAME
                        output directory


## USAGE
    In the Following lines I include some examples on how to use the package from your docker with the necessary 
    parameters. The package is already installed inside the Container (all the  commands are included in the Dockerfile 
    to simplify the usage)
    It's important to place all your pretrained models in the same folder and them as a volume inside the container 
    (using the -v option as in the examples). If you train new models, they would be saved in the out_dir_models folder.

    - Train Model 1
        
        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest train --modelnumber=1 --n_epoch=10 --train_data="data/dataset_single_train.csv" --val_data="data/dataset_single_test.csv" --out_dir_models=models --batch_size=8 --lr=0.001 --out_model_name="model1_test.save"

        Please replace [[datapath]] and [[modelpath]]: with the full paths on where to find the datasets and the models in your local computer

        For example: docker run -it --gpus all -v /home/facundo/PycharmProjects/servier/data/:/workspace/servier/data/ -v /home/facundo/PycharmProjects/servier/models/:/workspace/servier/models/   servier:latest train --modelnumber=1 --n_epoch=10 --train_data="data/dataset_single_train.csv" --val_data="data/dataset_single_test.csv" --out_dir_models=models --batch_size=8 --lr=0.001 --out_model_name="model1_test.save"
    
    
    - Train Model 2
    
        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest train --modelnumber=1 --n_epoch=10 --train_data="data/dataset_single_train.csv" --val_data="data/dataset_single_test.csv" --out_dir_models=models --batch_size=8 --lr=0.001 --out_model_name="model2_test.save"

    - Train Model 3
    
        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest train --modelnumber=3 --n_epoch=10 --train_data="data/dataset_multi_train.csv" --val_data="data/dataset_multi_test.csv" --out_dir_models=models --batch_size=8 --lr=0.001 --out_model_name="model3_test.save"

    - Evaluate Model 1

        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest evaluate --modelnumber=1  --modelpath=models/model1_final.save  --val_data="data/dataset_single_test.csv" --out_file_results="results/output1.csv"
    
    - Evaluate Model 2

        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest evaluate --modelnumber=2  --modelpath=models/model2_final.save  --val_data="data/dataset_single_test.csv" --out_file_results="results/output2.csv"

    - Evaluate Model 3

        docker run -it --gpus all -v [[datapath]]:/workspace/servier/data/ -v [[modelpath]]:/workspace/servier/models/   servier:latest evaluate --modelnumber=3  --modelpath=models/model3_final.save  --val_data="data/dataset_multi_test.csv" --out_file_results="results/output3.csv"


    - Predict Model 1

        docker run -it --gpus all   -v [[modelpath]]:/workspace/servier/models/   servier:latest predict --modelnumber=1 --modelpath="models/model1_final.save"  

    - Predict Model 2
    
        docker run -it --gpus all   -v [[modelpath]]:/workspace/servier/models/  servier:latest predict --modelnumber=2 --modelpath=models/model2_final.save

    - Predict Model 3

        docker run -it --gpus all   -v [[modelpath]]:/workspace/servier/models/  servier:latest predict --modelnumber=3 --modelpath=models/model3_final.save


## PACKING
    The python setuptools build and install are already generated in the Docker Dockerfile and generated when building 
    the docker image.  

## DATASETS

    In order to train the three models I created 3 train datasets and 3 Validation datasets. 
    Due to the number of records I decided to use 500 randomly selected records for the validation set 
    (called dataset_multi_test.csv and dataset_single_train.csv)
    For the rest of the records (4499), as the data set was not balanced for the binary classification problem (P1), 
    I decided to balance it creating a new dataset for the dataset_single_train.csv. This new dataset has the same 
    number of records for P1=1 than the previous dataset (dataset_single.txt), but the records with P1=0 where included 
    3 times to balance the full dataset. Another way of doping this would be stratifying each batch into the Neural 
    Network.
    For the mutl target binary classification, the train dataset has 4499 records and no augmentation was done but I 
    kept the same formating to use the same training loop.


## DOCKER
    
    The Docker is created with the included Dockerfile and I started from a pytorch image in order to make the 
    instalation easy.

    Nevertheless, I could not create a conda environment from the Dockerfile as included in the instructions, so I use
    the base environment inside the container to install al lthe necessary packages and libraries and facilitate the 
    instalation using the Dockerfile.


## MODEL CREATION 

    In the following lines I'll explain my reasoning for creating the deep learning models.
    If you see the file (`src/models.py`)  you will see two models, and Embedings + LSTM Model and a 
    LinearClassification Model using Linear layers + Dropout + BatchNorm. 
    The Linear Classification Model was used to set up the base for each of the 3 Models. 
    After this base created I went into a better model to predict this type of data and using Embedings + a Recurrent Layer turned out to be 
    a better approach and gave me a lower loss, better accurracy results and better F1 Score's results. 
    Note that the hidden layer of the lstm layer is preserved and feed into the next batch to increase the learning
    capabilities. 
    In any way I left the two models for future porpuses. 
    Note: In the training loop I included all the validation set accuracy measures to try to reduce the overfitting 
    of the model and try to learn at the same pace both in the training set and in the validation set due. 
    
## DATASETS 

    I created two Dataset Classes. 
    SmilesDataset: Deals withthe traiing o the model and the full evaluation of the validation set.
        - For Model 1, the featurization of the Smiles string is done using the rdkit, which delivers a 2048 array
          to be inserted in to the Depp Learning Model.
        - For Model 2 and 3 the full SMILE's string  yielding tokens which were used to generate a Graph. This Grpah is 
          converted it into a 2048 numpy array to be inputed into the LSTM Model. It would be interesting to see the 
          usage of a Graph Deep Learning Model  with this type of data, but it is out of the scope of the time given to 
          finish the project. 
    
    SmilesDataset_singleline: It is used to generate the prediction of a single smiles line for all 3 models.

## FLASK APP
    
    Run your Docker to publish your model and be available at port 6000. It will use model1_final.save provided in the zip file. 
    
    
    docker run -it --gpus all  -p 7000:7000 -v [[modelpath]]:/workspace/servier/models/   servier:latest /bin/bash -c "FLASK_ENV=development FLASK_APP=app.py flask run --host 0.0.0.0 -p 7000"
    For ex:
    docker run -it --gpus all  -p 7000:7000 -v /home/facundo/PycharmProjects/servier/models/:/workspace/servier/models/   servier:latest /bin/bash -c "FLASK_ENV=development FLASK_APP=app.py flask run --host 0.0.0.0 -p 7000"


    Open a new terminal an execute 'python request.py' (the request file in located in the src directory, the request package must  be installed).
    in order to get a predisction for an inputed SMILES string.
    
    

## RESULTS
    
    In the following lines I attach the best model performances I had for each of the 3 models I trained.
    
    - Model 1:
        file:model1_final.save
         * Traning Set:

            Confusion Matrix P1
            [[3176    0]
             [   0 3705]]
            Classification report
                          precision    recall  f1-score   support
            
                       0       1.00      1.00      1.00      3176
                       1       1.00      1.00      1.00      3705
            
                accuracy                           1.00      6881
               macro avg       1.00      1.00      1.00      6881
            weighted avg       1.00      1.00      1.00      6881

   

        * Validation set:
            Confusion Matrix P1
            [[ 19  78]
             [ 52 351]]
            Classification report
                          precision    recall  f1-score   support
            
                       0       0.27      0.20      0.23        97
                       1       0.82      0.87      0.84       403
            
                accuracy                           0.74       500
               macro avg       0.54      0.53      0.53       500
            weighted avg       0.71      0.74      0.72       500

    - Model 2:
        * Traning Set:
        Confusion Matrix P1
        [[3172    4]
         [  32 3673]]
        Classification report
                      precision    recall  f1-score   support
        
                   0       0.99      1.00      0.99      3176
                   1       1.00      0.99      1.00      3705
        
            accuracy                           0.99      6881
           macro avg       0.99      1.00      0.99      6881
        weighted avg       0.99      0.99      0.99      6881


        * Validation set:
            Confusion Matrix P1
            [[ 18  79]
             [ 74 329]]
            Classification report
                          precision    recall  f1-score   support
            
                       0       0.20      0.19      0.19        97
                       1       0.81      0.82      0.81       403
            
                accuracy                           0.69       500
               macro avg       0.50      0.50      0.50       500
            weighted avg       0.69      0.69      0.69       500


    - Model 3:
        * Traning Set:
        Confusion Matrix P1-P9
            [[[ 112  682]
              [  26 3679]]
            
             [[ 109  663]
              [  16 3711]]
            
             [[ 140  592]
              [  31 3736]]
            
             [[  80  684]
              [  22 3713]]
            
             [[ 163  579]
              [  23 3734]]
            
             [[ 175  524]
              [  63 3737]]
            
             [[  74  631]
              [  10 3784]]
            
             [[  88  621]
              [  23 3767]]
            
             [[  86  636]
              [  10 3767]]]
            Classification report
                          precision    recall  f1-score   support
            
                       0       0.84      0.99      0.91      3705
                       1       0.85      1.00      0.92      3727
                       2       0.86      0.99      0.92      3767
                       3       0.84      0.99      0.91      3735
                       4       0.87      0.99      0.93      3757
                       5       0.88      0.98      0.93      3800
                       6       0.86      1.00      0.92      3794
                       7       0.86      0.99      0.92      3790
                       8       0.86      1.00      0.92      3777
            
               micro avg       0.86      0.99      0.92     33852
               macro avg       0.86      0.99      0.92     33852
            weighted avg       0.86      0.99      0.92     33852
             samples avg       0.86      0.99      0.91     33852
                  
   
        * Validation set:
            Confusion Matrix P1-P9
            [[[  4  93]
              [ 14 389]]
            
             [[  1  87]
              [  6 406]]
            
             [[  2  86]
              [ 16 396]]
            
             [[  1  83]
              [  9 407]]
            
             [[  3 101]
              [ 10 386]]
            
             [[ 10  66]
              [ 25 399]]
            
             [[  1  97]
              [  7 395]]
            
             [[  2  86]
              [ 11 401]]
            
             [[  2  93]
              [  5 400]]]
            Classification report
                          precision    recall  f1-score   support
            
                       0       0.81      0.97      0.88       403
                       1       0.82      0.99      0.90       412
                       2       0.82      0.96      0.89       412
                       3       0.83      0.98      0.90       416
                       4       0.79      0.97      0.87       396
                       5       0.86      0.94      0.90       424
                       6       0.80      0.98      0.88       402
                       7       0.82      0.97      0.89       412
                       8       0.81      0.99      0.89       405
            
               micro avg       0.82      0.97      0.89      3682
               macro avg       0.82      0.97      0.89      3682
            weighted avg       0.82      0.97      0.89      3682
             samples avg       0.82      0.97      0.88      3682

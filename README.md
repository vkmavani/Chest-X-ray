# Chest-X-ray

* [ChestX-ray8 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) which contains 108,948 frontal-view X-ray images of 32,717 unique patients.

  -> Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions
      
  -> Here, in this project I created model which predict 'positive' or 'negative' for each of the pathologies

* In this particualr notebook I used ~1000 images subset due to lack of resources. This data is provided by Coursera.
* They also provided processed labels for our small sample and generated three new files to get started. These three files are:
    1. `train-small.csv`: 875 images from our dataset to be used for training.
    2. `valid-small.csv`: 109 images from our dataset to be used for validation.
    3. `test.csv`: 420 images from our dataset to be used for testing.
  
* This dataset has been annotated by consensus among four different radiologists for 5 of our 14 pathologies:
    - `Consolidation`
    - `Edema`
    - `Effusion`
    - `Cardiomegaly`
    - `Atelectasis`

* In this project I also check whether there is leakage between two datasets, and also used weighted loss

* Due to lake of GPU I have to use Pre-Trained Model
      
     [Densenet.hdf5](https://drive.google.com/file/d/17pZaEJ_7s6NPC79ln227SDj-J_9vz3u3/view?usp=sharing)
     [Pre-trained Model](https://drive.google.com/file/d/1jKmqbnioUXWHD0ThVMuhPcupoIj6hVBt/view?usp=sharing)

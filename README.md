# Multimodal Machine learning for classification of respiratory illness
This repository consists of project files belonging to Machine learning research lab winter semester 2021-2022 at University of Koblenz and Landau.

# About
In healthcare, multimodal machine learning combines data from several modalities such as image, text, and audio. Because the feature vector now contains information from many data modalities, this fusion improves prediction accuracy. The fusion procedures can be divided into two categories: early fusion and late fusion. A notable research work [1] aimed to perform multimodal fusion on chest X-ray images and corresponding radiology report to predict respiratory illness in the incoming data sample. This research work is  chosen as our baseline approach. The image feature vectors extracted from raw chest x-ray images and text feature vectors extracted from corresponding radiology reports are passed through four state-of-the-art Visual Language models like VisualBERT, LXMERT, UNITER, PixelBERT to perform multi-class classification of respiratory illness. This research work aims at enhancing the results of the visual language models VisualBERT, LXMERT,UNITER by adding an additional layer of late fusion which combines the output feature vector of Visual language model with the output feature vector of another powerful NLP model trained only on radiology text reports. This combined feature vector is further examined by the neural network before providing the final prediction.

# Datasets 
* MIMIC-CXR 
    * MIMIC-CXR is a huge dataset of 227835 radiology text reports and 377110 chest X-ray images that is freely available where every sample is classified into 13 respiratory findings. 
    * The radiology text report extarction details can be found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/MIMIC-CXR/extractTextReportsMIMIC.ipynb)
    * The image extraction details can be found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/MIMIC-CXR/extractImagesMIMIC.ipynb)
    * Data pre-processing of extracted radiology text reports can be found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/MIMIC-CXR/MIMIC_textReports_Preprocessing.ipynb)
    * 22000 randomly sampled image-text pairs are used for training.

* OpenI
    * OpenI dataset contains chest X-ray images and radiology text reports. The official [git repo](https://github.com/YIKUAN8/Transformers-VQA) has 3684 chest x-ray image and radiology pairs, each of which is linked to one of 15 respiratory illnesses. These data samples are taken for training the models.

# Usage

## Dependencies
  * Clone the git repository in your local system
  * Python version : 3.8.10
  * Install the required dependencies using the command - pip install requirements.txt
      

## Visual-Language models

* VisualBERT 
  * The VisualBERT code run on OpenI dataset is found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/VisualBERT/openI/VisualBERT_OpenI.ipynb). Original implementation is found in the [git repo](https://github.com/YIKUAN8/Transformers-VQA) 
  * VisualBERT code run on MIMIC-CXR dataset can be found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/VisualBERT/MIMIC/VisualBERT_MIMIC__Testing.ipynb)

* UNITER
  * UNITER implementation details on openI dataset is found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/Uniter/OpenI/UNITER_OpenI__Testing.ipynb)
  * UNITER implementation details on MIMIC dataset is found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/Uniter/MIMIC/UNITER_MIMIC__Testing.ipynb)

* LXMERT
  * LXMERT implementation details on openI dataset is found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/LXMERT/OpenI/LXMERT_OpenI_Testing.ipynb)
  * LXMERT implementation details on MIMIC dataset is found [here](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/LXMERT/MIMIC/LXMERT_MIMIC__Testing.ipynb)

## Natural Language Processing models

* Several word embedding models were run on openI dataset to choose the embedding which best suits our dataset. The combination of Glove word embedding with Bi-directional GRU and attension layer gave the best classification results The implementation details are found below
  * [BERT](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/NLP/openI/OpenI_BERT.ipynb) 
  * [Doc2Vec](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/NLP/openI/OpenI_Doc2Vec.ipynb)
  * [Glove](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/NLP/openI/OpenI_GloVe_Attention.ipynb)

## Image processing and feature extraction
* Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. We are using it in our application to extract features, boxes from the given X-ray image.
  * [Detectron2](https://github.com/airsplay/py-bottom-up-attention)



## Late Fusion
Late fusion of predictions from visual language models VisualBERT, LXMERT,UNITER and predictions from corresponding NLP models were performed. The results can be found below
* Results obtained on OpenI dataset
  * [VisulBERT+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/VisualBERT/openI/LateFusion_VisualBERT_OpenI.ipynb)
  * [UNITER+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/Uniter/OpenI/LateFusionFinalOutput__UNITER_OpenI.ipynb)
  * [LXMERT+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/LXMERT/OpenI/LXMERT_OpenI_Testing.ipynb) 
* Results obtained on MIMIC CXR dataset
  * [VisulBERT+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/VisualBERT/MIMIC/LateFusion_VisualBERT_MIMIC.ipynb)
  * [UNITER+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/Uniter/MIMIC/LateFusionFinalOutput__UNITER_MIMIC.ipynb)
  * [LXMERT+NLP](https://github.com/Pavithree/ResearchLab_MultimodalFusion/blob/main/notebooks/LXMERT/MIMIC/LateFusion_LXMERT_MIMIC.ipynb) 

## Results

* OpenI
  * VisualBERT's results (AUC) have improved by 1.21 percent, UNITER's results have improved by 1.24 percent, and LXMERT's results have remained unchanged with our proposed strategy.

* MIMIC-CXR
  * The results obtained by incorporating our proposed technique demonstrate a significant improvement with respect to precision, recall and AUC for MIMIC dataset.

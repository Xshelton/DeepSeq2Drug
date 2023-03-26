# DeepSeq2Drug
## This is the code repository of DeepSeq2drug. 
## It contains three main parts. 
* 1. Feature/Embedding Generation
* 2. Model Validation
* 3. Case Study Prediction

* 1.	For Feature/Embedding Generation, the input files are too large for the pre-processing to be directly uploaded to GitHub; we will upload them to our server in the future.
* 2.	The Model Validation needs features/embeddings generated above and also constructed embedding dataset to train/validate the models.
* 3.	For the Case Study prediction, we already built/trained the models. Users need to generate corresponding features from your case and then run the source code to get the corresponding predictive results. we already built the software for Case Study prediction, the community version can be found at http://deepseq2drug.cs.cityu.edu.hk/software-download/


#### We build the code based on the folder of features/embeddings.

#### folder name *files descriptions
#### 0_drug_feature_pool  *Drug feature files from different feature extractors
#### 0_virus_feature_pool  *Virus sequence/ NLP-Based features files from different feature extractors
#### constructed dataset *Pair-wised relations between virus and drugs (original dataset pairs+sampling negative pairs)
#### Embedding dataset *using drug/virus embedding and constructed dataset to build constructed embedding dataset

#### How to validate the quality of embedding:
put drug features/embedding into the 0_drug_feature_pool  
put virus features/embedding into the 0_virus_feature_pool  
Download original dataset  
original dataset, features samples can be download at:
https://drive.google.com/file/d/1mbp3d88mtBzdo2hVF5E7mVvgJV7haLZd/view?usp=sharing
run Step1-dataset_construct.py to generate constructed dataset/  
or download constructed dataset/   
run Step2_generate_ 5folds_files.py
run Step3- RF-5folds-validate-upload.py   
it will automatically generate all the folders needed for trainning and validation.  



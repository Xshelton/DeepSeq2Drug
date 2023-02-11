# DeepSeq2Drug
## This is the code repository of DeepSeq2drug. 
## It contains three main parts. 
* 1. Feature/Embedding Generation
* 2. Model Validation
* 3. Case Study Prediction

* 1.	For Feature/Embedding Generation, the input files are too large for the pre-processing to be directly uploaded to GitHub; we will upload them to our server in the future.
* 2.	The Model Validation needs features/embeddings generated above and also constructed embedding dataset to train/validate the models.
* 3.	For the Case Study prediction, we already built/trained the models. Users need to generate corresponding features from your case and then run the source code to get the corresponding predictive results. 


#### We build the code based on the folder of features/embeddings.

#### folder name *files descriptions
#### 0_drug_feature_pool  *Drug feature files from different feature extractors
#### 0_virus_feature_pool  *Virus sequence/ NLP-Based features files from different feature extractors
#### 0_case_study_feature_pool * Sequence/ NLP-Based features for novel virus
#### constructed dataset *Pair-wised relations between virus and drugs (original dataset pairs+sampling negative pairs)
#### constructed embedding dataset *using drug/virus embedding and constructed dataset to build constructed embedding dataset

#### Tips for conducting a new case study:
* Step1:  
Run Step2.py to generate constructed embedding dataset. (if it existed, skip this step)
* Step2:  
Run the Feature extractor to generate sequence-based features/embeddings and NLP-base embeddings.
* Step3:  
Put results of Step2 into 0_case_study_feature_pool 
* Step4:  
Run Step4.py to generate corresponding models( using files from 0_drug_feature_pool/0_virus_feature_pool/ constructed embedding dataset)
* Step 5:  
Running step7.py/step8.py to generate scores for the specific cases will generate a folder named “{your new case}Case_study_prediction”.
Put “DBID_and_name(case).csv” into the folder and run “Results_extractor.py.”, it will generate “{your new case}_all_score_FTW_top20.csv”, showing the top-20 predictive results of DeepSeq2Drug.

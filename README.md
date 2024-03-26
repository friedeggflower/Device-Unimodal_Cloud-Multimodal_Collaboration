# Device-Unimodal_Cloud-Multimodal_Collaboration
Codes for &lt;Device-Unimodal Cloud-Multimodal Collaboration>.  
For the experiment on the public dataset, both data and codes are provided.  

## Experiments on public dataset
1. Data preprocess  
(1) Download the Fashion-Gen dataset to 'Public_Dataset/Data'  
[fashiongen_256_256_train.h5](https://drive.google.com/file/d/1yR-_NJ6CxCYB2i8UECEB2doNssj5J8Li/view?usp=sharing)  
[fashiongen_256_256_validation.h5](https://drive.google.com/file/d/1R4JtSAtbbOi9mByBj-KpchNf9nDn6BO1/view?usp=sharing)   
(2) Data preprocess  
   ```bash
   # Load data from h5.file.  
   $ python Public_Dataset/data_preprocess/get_h5.py
   # Split training and testing datasets.
   $ python Public_Dataset/data_preprocess/split_data.py
   ```  

2. Device-side  
The codes of Faster-RCNN mainly refers to [the Pytorch implementaion of bottom-up-attention models](https://github.com/MILVLG/bottom-up-attention.pytorch), including the pretrained model.  
(1) Dependency
   - PyTorch 1.8.1
   - Detectron2
   - OpenCV
   
   To install Detectron2, we recommend:  
      ```bash
      $ git clone --recursive https://github.com/MILVLG/bottom-up-attention.pytorch
      $ cd detectron2
      $ pip install -e .
      ```

   Download the pretrained ResNet50 model from [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EfYoinBHrFlKmKonocse8yEBXN-hyCHNygYqjxGpIBsPvQ?download=1) to 'Public_Dataset/Device_unimodal/checkpoint/baseline/'   

   (2) Adaptation of the pretrained visual model and testing
   ```bash
   $ python Public_Dataset/Device_unimodal/Device_step1.py
   ```   

   (3) Device-side training and testing
   ```bash
   $ python Public_Dataset/Device_unimodal/Device_step2.py
   ```  

   (4) Extract visual features for the cloud and save in '.npz' format  
   ```bash
   # Unified visual features.  
   $ python Public_Dataset/Device_unimodal/visual_fea_uni.py
   # Personalized visual features.  
   $ python Public_Dataset/Device_unimodal/visual_fea_per.py
   ```    

3. Cloud-side   
The codes of Uniter mainly refers to [UNITER](https://github.com/ChenRocks/UNITER), including the pretrained model.    
   (1) Dependency   
   - apex
   - lmdb
   - nltk (including wordnet)

   ```bash
   $ pip install -r Public_Dataset/Cloud_multimodal/requirements.txt
   ``` 
   Download the pretrained UNITER model to 'Public_Dataset/Cloud_multimodal/checkpoint/pretrained'
   ```bash
   $ bash Public_Dataset/Cloud_multimodal/scripts/download_pretrained.sh 
   ```    

   (2) Prepare dataset into lmdb  
   Prepare textual inputs. (Mask possible category-related words in texts of samples. We use the nltk package to find out all possible related words. 
   ```bash
   $ python Public_Dataset/Cloud_multimodal/txtdb.py
   ```    
   Prepare visual inputs.  
   ```bash
   $ python Public_Dataset/Cloud_multimodal/imgdb.py
   ```  

   (3) Adaptation of the pretrained multimodal model and testing
   ```bash
   $ python Public_Dataset/Cloud_multimodal/Cloud_step1.py
   ```   

   (4) Train and test for TuneUni and PromptConcat
   ```bash
   # TuneUni
   $ python Public_Dataset/Cloud_multimodal/Cloud_step3_TuneUni.py
   # PromptConcat
   $ python Public_Dataset/Cloud_multimodal/Cloud_step3_Prompt.py
   ```    

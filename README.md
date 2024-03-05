# Handwritten Math Symbol and Digit Recognitions Project ✍️
#### HUDK4054001 Individual Assignment 2
-------------
## Table of Contents
- [Project Details](#ProjectDetails)
- [Project Objectives and Goals](#ProjectObjectivesandGoalss)
- [About the Dataset](#AbouttheDataset)
- [Methodology of the Project](#Methodology)
- [Data Preparation](#DataPreparation)
- [Data Folder Labels](#DataFolderLabel)
- [Import Necessary Libraries](#ImportNecessaryLibraries)
- [Models and Result](#Models)
- [Data Storage/Access Information](#DataStorage)
- [License](#License)
- [Self Reflection of Creating Metadata](#SelfReflectionofMetadata)
- [Feedback From Course](#Feedback)

-------------
## 1️⃣ Project Details <a name="ProjectDetails"></a>
Our project, which focuses on recognizing students' handwritten digits, operators, and other basic mathematical symbols, serves as a foundational step towards developing more advanced models in the field of educational technology.

- **Project Name:** Handwritten Math Symbol and Digit Recognition  
- **Course:** HUDK4050 Core Methods Education Data Mining (2023 Fall)
- **Data Manager:** Simeng Zhao
  - **ORCID ID:** 0009-0000-3577-6114
  - **Email:** sz3163@tc.columbia.edu
- **Group Members:** Summer Wu, Viola Tan, Theresa Zhao, Yanfei Chen, Bryan Wu, Carla Hounshell, Simeng Zhao 
- **Professor:** Dr. J.D Jayaraman (jj2797@tc.columbia.edu)
- **Affiliation:** Teachers College, Columbia University
-------------
## 2️⃣ Project Objectives and Goals <a name="ProjectObjectivesandGoalss"></a>
- **Focus:** Recognizing students' handwritten digits, operators, and basic mathematical symbols  
- **Purpose:** Foundation for future advancements in educational technology  
- **Applications:** OCR technology in education, particularly in mathematics, offers several advantages. It enhances learning through interactive apps, provides immediate feedback on handwritten work, streamlines grading processes, and supports personalized instruction. Additionally, OCR serves as assistive technology for students with learning disabilities, promoting inclusivity. Overall, OCR integration improves educational experiences, efficiency, and inclusivity in mathematics education.
-------------
## 3️⃣ About the Dataset <a name="AbouttheDataset"></a>

- **Name:** Handwritten math symbol and digit dataset
- **Source:** [Kaggle Handwritten Math Symbol Dataset](https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset/data)
  - *FYI: You don't need to sign up to download the dataset from Kaggle* 😊
- **Creator:** Clarence Zhao (Owner)
- **Year of Creation:** 2019
- **Origin:** Zhao meticulously curated this dataset using a stylus on an iPad, involving the handwriting of three individuals.
- **Composition:** It includes images of 10 mathematical digits ranging from 0 to 9 and six mathematical operators (plus, minus, times, divide, equal, and decimal), resulting in 16 distinct classes.
- **Composition Process:** Each participant created multiple pages featuring handwritten digits and operators, subsequently composed by cropping individual pages.
- **Contents:**
  - **Images:** The dataset contains over 8500 handwritten symbol images. All images are in JPG format and are 155x155 pixels in size.
  - **Labels:** Each image is associated with a label indicating the corresponding mathematical symbol it represents.
- **File Structure:**
  - **training set**: Folder containing the training set images.
  - **validation set**: Folder containing the validation/evaluation set images.

-------------
## 4️⃣ Methodology of the Project <a name="Methodology"></a>
- **Approach**: Employ machine learning to recognize handwritten digits, operators, and mathematical symbols.
- **Data Source**: Utilize a dataset crafted by Clarence Zhao on Kaggle, containing over 8500 handwritten symbol images.
- **Data Collection**: Zhao curated the dataset using a stylus on an iPad, involving three individuals to create images of digits and operators.
- **Model Development**: Develop a machine learning model, possibly using CNNs, trained on labeled images to learn patterns.
- **Evaluation**: Assess model performance on a validation set using metrics like accuracy, precision, and recall.
- **Improvement**: Iterate on the model through techniques like hyperparameter tuning for optimization.
- **Deployment**: Deploy the trained model in educational technology applications for enhanced learning experiences.
-------------

## 5️⃣ Data Preparation <a name="DataPreparation"></a>
- **Dataset Reduction and Preprocessing:**
In the dataset, there are 20 folders for training and 18 folders for evaluation. During the data cleaning process, we removed the "sign" and "number" folders from the evaluation set. Additionally, we excluded folders labeled "original number", "original sign", "other number", and "other sign" from both the training and evaluation sets.
-------------
## 6️⃣ Data Folder Labels <a name="DataFolderLabel"></a>
| Folder Name        | Meaning                                           |
|--------------------|-------------------------------------------------------|
| eval               | Testing data                                          |
| train              | Training data                                         |

#### ‘eval' Folder:
| Folder Name   | Meaning                                                 |
|---------------|---------------------------------------------------------|
| decimal val   | Individual data images for the decimal sign             |
| div val       | Individual data images for the division sign            |
| eight         | Individual data images for the number eight             |
| equal val     | Individual data images for the equal signs              |
| five          | Individual data images for the number five              |
| four          | Individual data images for the number four              |
| minus val     | Individual data images for the minus signs              |
| nine          | Individual data images for the number nine              |
| one           | Individual data images for the number one               |
| plus val      | Individual data images for the plus signs               |
| seven         | Individual data images for the number seven             |
| six           | Individual data images for the number six               |
| three         | Individual data images for the number three             |
| times val     | Individual data images for the times signs              |
| two           | Individual data images for the number two               |
| zero          | Individual data images for the number zero              |

#### 'train' Folder:
| Folder Name       | Meaning                                                 |
|-------------------|---------------------------------------------------------|
| decimal           | Individual data images for the decimal signs            |
| div               | Individual data images for the division signs           |
| eight             | Individual data images for the number eight             |
| equal             | Individual data images for the equal signs              |
| five              | Individual data images for the number five              |
| four              | Individual data images for the number four              |
| minus             | Individual data images for the minus sign               |
| nine              | Individual data images for the number nine              |
| one               | Individual data images for the number one               |
| plus cleaned      | Individual data images for the plus signs               |
| seven             | Individual data images for the number seven             |
| six               | Individual data images for the number six               |
| three             | Individual data images for the number three             |
| times             | Individual data images for the times sign               |
| two               | Individual data images for the number two               |
| zero              | Individual data images for the number zero              |

 
**Distribution of classes in our datasets:**
```python
(array(['decimal', 'div', 'eight', 'equal', 'five', 'four', 'minus',
       'nine', 'one', 'plus', 'seven', 'six', 'three', 'times', 'two',
       'zero'], dtype='<U7'), array([513, 544, 429, 554, 431, 431, 549, 430, 432, 545, 430, 429, 429,
       555, 430, 426]))
(array(['decimal', 'div', 'eight', 'equal', 'five', 'four', 'minus',
       'nine', 'one', 'plus', 'seven', 'six', 'three', 'times', 'two',
       'zero'], dtype='<U7'), array([76, 78, 54, 80, 54, 54, 80, 54, 55, 78, 54, 53, 54, 80, 54, 52]))
```

- **Image Standardization:**
  - Resized all images to 100x100 pixels for consistency.
  - Simplified model training without compromising data integrity


- **Color Format Retention:**
  - Preserved images in original color format (black ink on white background).
  - Accurately represented visual characteristics, aiding model accuracy.

- **Label Encoding:**
  - Transformed categorical labels into numerical form using `LabelEncoder`.
  - Facilitated numerical processing and preserved class mappings.
```python 
label_encoder = preprocessing.LabelEncoder()
train_label_encoded = label_encoder.fit_transform(train_label)
test_label_encoded = label_encoder.fit_transform(test_label)

y_train = to_categorical(train_label_encoded, 16)
y_test = to_categorical(test_label_encoded, 16)
```

-------------                         

## 7️⃣ Import Necessary Libraries <a name="ImportNecessaryLibraries"></a>

```python
import os
from os import listdir
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import sklearn as sk
from sklearn import preprocessing
from keras.utils import to_categorical
from tensorflow import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
```


-------------
## 8️⃣ Models and Result <a name="Models"></a>

#### 1. Fully Connected Models:
- Fully connected models process flattened input data through multiple layers where each neuron is connected to every neuron in the next layer, commonly used for image classification. 


#### 2. Convolutional Neural Networks (CNNs):
- Convolutional Neural Networks (CNNs) are specialized models for processing grid-like data, such as images, preserving spatial structure using convolutional and pooling layers, and commonly used for image classification tasks.

#### Comparision of Model Accuracy
<img width="181" alt="截屏2024-02-19 下午11 14 08" src="https://github.com/SimengZhao/Handwritten-Letter/assets/160539675/fc72dd88-dbdd-4110-969a-4d356b0d0d4f">

Our proprietary CNN model demonstrated impressive accuracy, reaching a notable high score of 0.9458, while our dense model achieved a respectable accuracy level of 0.7079. Both models performed admirably, as indicated by these results. However, it was observed that pretrained models, particularly VGG16, outperformed our models in terms of accuracy, achieving exceptionally high scores. This highlights the strength of pretrained networks. Our experience with transfer learning provided valuable insights into the complexities of CNN models. Given the intricate nature of these models, we carefully adjusted the number of epochs to optimize performance without prolonging execution times excessively. This approach enabled us to strike a balance between efficiency and effectiveness in our model training process.

#### Conclusion and Reflection
Our project aimed to explore the fusion of technology and machine learning in education. It's important to recognize the project's limitations: we focused on recognizing specific math symbols, whereas real-world educational scenarios often involve more complex symbols and text. Additionally, our understanding of machine learning was still developing, so our project mainly laid the groundwork for more advanced applications.

-------------
## 9️⃣ Data Storage/Access Information <a name="DataStorage"></a>

Our project utilized Google Colab for its collaborative and cloud-based computing environment. This platform allowed us to seamlessly work together on our project, leveraging its resources for data processing, model development, and analysis.

#### Publication Status:

Throughout the project, we conducted presentations to showcase our progress and findings. As our work is primarily for a course assignment and involves shared collaboration, we haven't yet published our results in a formal academic or research setting. However, our focus has been on rigorously exploring and understanding the problem domain, experimenting with various methodologies, and synthesizing our findings into actionable insights.

Using Colab as our project platform has facilitated efficient collaboration, enabling us to leverage its resources and tools for data analysis, model training, and result visualization. While our results have been presented internally, future plans may involve further refinement of our models and potentially disseminating our findings through publication or presentation in relevant forums.


-------------
## 🔟 License <a name="License"></a>
Copyrighted by Original Authors

*Note: It's important to note that without explicit licensing terms, the dataset may not be freely available for use, modification, or distribution, and permission from the original authors may be required.*

-------------
## Self Reflection of Creating Metadata <a name="SelfReflectionofMetadata"></a>
Accoriding to the [six types of Metadata](https://atlan.com/what-is-metadata/#examples-of-metadata) that Dr. Yasemin Gulbahar shared in class, my project metadata encompasses technical details, governance information with Clarence Zhao credited as the dataset creator, operational aspects detailing data flow from collection to model deployment, and quality metrics for model evaluation. Collaboration is implied through the mention of group members and a professor. Additionally, example usage of code snippets suggests practical applications for the dataset and models. This comprehensive approach provides viewers/users with a clear understanding of the project's scope, context, and potential applications.


**🔹 Which metadata standard did you choose and why?**
I chose the DDI Codebook (DDI-C) standard for my metadata because it fits well with my need for simplicity and individual dataset documentation. As I am very new to GitHub and metadata, DDI-C's straightforward structure, based on XML, would be a user-friendly approach to effectively documenting my dataset.

**🔹 Which template/software did you use?**
I used GitHub to write the README-style metadata because I had never used it before and it seemed like a platform where I could have a lot of fun exploring and creating.

**🔹 What was the most challenging part of creating a ReadME file? How did you overcome these obstacles?**
The most challenging part of creating the README file was getting started, especially since I had never used GitHub before. It took some time to familiarize myself with the platform and its features. Additionally, I encountered difficulties due to the lack of information provided in the original dataset. There was no contact information, license details, DOI, or a comprehensive description of the dataset folders, which made it challenging to provide thorough metadata. To overcome these obstacles, I relied on online resources, some tutorials, and the GitHub guidance.

## *Feedback From Course* <a name="Feedback"></a>
1. If more than one group member, then provide their roles
2. the platform you are using
3. Provide the file versions (and you should have it since we have cleaned it by deleting it from the original files)
4. Provide visuals (your process line, your progress as a picture is preferred)

<br>


#### Congratulation you've made it through reading my ReadMe. Here is your free e-cookie to enjoy! 
<img src="https://github.com/SimengZhao/Handwritten-Letter/assets/160539675/11ce83ca-9c4a-4542-9d79-38962448e6f4" width="400">

[source of the cookie picture](https://www.123rf.com/clipart-vector/chocolate_chip_cookie.html)

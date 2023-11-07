# Alzheimer_Ml_Project

# An efficient Alzheimer’s Disease Diagnosis with fine-tuned Transfer Learning models

<div style="display: flex; align-items: center; justify-content: center;">
  <div style="flex: 1;">
    <div align="center"><img src="https://i.imgur.com/Y91RCPj.png" width=350 height=350></div>
  </div>
  <div style="flex: 1;">
    <svg width="350" height="350" xmlns="http://www.w3.org/2000/svg">
      <!-- Your SVG code goes here --><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><g data-name="Doctor Man"><path d="M26.75 30a.755.755 0 0 1-.75.75v-1.5a.755.755 0 0 1 .75.75zM38 29.25v1.5a.75.75 0 0 1 0-1.5z" style="fill:#eac2b9"/><path d="M45 8v9.27a2 2 0 0 0-1-.27h-3v-4s-2 0-5.4-3c0 0-3.6 4-12.6 4v3h-3a2 2 0 0 0-1 .27V8a4 4 0 0 1 4-4 3 3 0 0 1 3-3h12a7 7 0 0 1 7 7z" style="fill:#494a59"/><path d="M46 18v3a2.006 2.006 0 0 1-2 2h-4.22a9.356 9.356 0 0 0 .22-2v-5h4a2.006 2.006 0 0 1 2 2zM24.22 23H20a2.006 2.006 0 0 1-2-2v-3a2.006 2.006 0 0 1 2-2h4v5a9.356 9.356 0 0 0 .22 2zM38 26.7V34l-4 5h-4l-4-5v-7.3a8.976 8.976 0 0 0 12 0z" style="fill:#eac2b9"/><path d="M53.283 39.877 38 33l-4 5h-4l-4-5-15.283 6.877A8 8 0 0 0 6 47.173V62.25a.75.75 0 0 0 .75.75h50.5a.75.75 0 0 0 .75-.75V47.173a8 8 0 0 0-4.717-7.296z" style="fill:#ffb64d"/><path d="M44 35.7v.08a17.958 17.958 0 0 1-4.84 12.28 9.857 9.857 0 0 1-14.32 0A17.958 17.958 0 0 1 20 35.78v-.08l6-2.7 4 5h4l4-5z" style="fill:#e6ecff"/><path d="M26 29.25v1.5h-3A7.255 7.255 0 0 0 15.75 38v6.6l2.67 1.78a.75.75 0 0 1 .33.62v2a.75.75 0 0 1-1.5 0v-1.6L15 45.9l-2.25 1.5V49a.75.75 0 0 1-1.5 0v-2a.75.75 0 0 1 .33-.62l2.67-1.78V38A8.76 8.76 0 0 1 23 29.25zM49.75 38v6a.75.75 0 0 1-1.5 0v-6A7.255 7.255 0 0 0 41 30.75h-3v-1.5h3A8.76 8.76 0 0 1 49.75 38z" style="fill:#4294ff"/><path style="fill:#d0dbf7" d="m30 38-5 5-5-7.3 6-2.7 4 5z"/><path style="fill:#343544" d="M37.64 41.64 35 46h-6l-2.64-4.36L30 38h4l3.64 3.64z"/><path style="fill:#d0dbf7" d="M44 35.7 39 43l-5-5 4-5 6 2.7z"/><path d="M36.32 50.16a9.989 9.989 0 0 1-8.64 0L29 46h6z" style="fill:#494a59"/><path d="M49.75 52v11h-1.5V52a.75.75 0 0 1 1.5 0zM15.75 52v11h-1.5V52a.75.75 0 0 1 1.5 0z" style="fill:#ea972a"/><path d="M23 13v8a9 9 0 0 0 9 9 9 9 0 0 0 9-9v-9s-2 0-5.4-3c0 0-3.6 4-12.6 4z" style="fill:#ffddd4"/><circle cx="49" cy="46" r="2.75" style="fill:#376cfb"/></g></svg>
      <rect width="350" height="350" fill="#FF5733"/>
    </svg>
  </div>
</div>


This work presents a reliable mechanism for early illness identification in the context of the worldwide healthcare difficulties brought on by Alzheimer’s disease. This work applies deep learning techniques using a large dataset of high-quality brain MRI images that includes people with Alzheimer’s disease and cognitively normal patients divided into 4 classes:

- Moderate Demented
  
- Mild Demented
  
- Very Mild Demented
  
- Non Demented

To create an automated diagnostic system, this work specifically use four transfer learning architectures—EfficientNetV2B3, InceptionNetV3, NASNetMobile and MobileNetV2 as base models. EfficientNetV2B3 which underwent thorough training and evaluation and was directed by important measures like loss and accuracy, ultimately showing a diagnostic accuracy of 96.72 %.

This dataset and model combination, supported by the Weights and Biases platform for real-time monitoring and metric logging, hold promise in advancing early Alzheimer’s disease detection and improving patient outcomes, addressing the significant societal and healthcare challenges
related to this condition. 

## Alzheimer's Disease and the importance of its early detection

Alzheimer's disease is a devastating and incurable neurodegenerative disorder that primarily affects the elderly, although early-onset cases can occur as well. It is the most common cause of dementia, responsible for significant cognitive and functional decline, and has a profound impact on individuals and their families. 

### Disease Prevalence

Alzheimer's disease is a widespread health concern, with an increasing global prevalence due to an aging population. The Alzheimer's Association estimates that over 6 million Americans are living with Alzheimer's disease, and this number is projected to rise significantly in the coming years. Worldwide, there are approximately 50 million people living with dementia, and this number is expected to triple by 2050.

### Timely Intervention

Early detection of Alzheimer's disease is vital because it allows for timely intervention. Although there is currently no cure for Alzheimer's, early diagnosis enables individuals to access available treatments and support services that can help manage the condition, alleviate symptoms, and improve the quality of life for both patients and their caregivers. As Alzheimer's disease is progressive, an early diagnosis helps individuals and their families with the opportunity to plan for the future.

## Dataset Description

The dataset used in this research work is titled as Alzheimer’s Dataset (S. Dubey). The research work was done on a total of aroung 6,400 images.
These are MRI images which are then further segregated into 4 classes. These 4 classes are Mild Demented, Moderate Demented, Non Demented and Very Mild Demented. Moderate Demented has 64 images, Mild Demented has 896 images, Very Mild Demented has 2240 images and Non Demented has 3200
images. The distribution of the dataset amongst these 4 classes is shown below:


<div align="center"><img src="https://i.imgur.com/SwlX1I4.png" width=500 height=350></div>

The subsequent figure displays a set of sample images from the dataset processed by setting color maps as ”jet”. The ”jet” colormap is a rainbow-like colormap that maps numerical values to colors, transitioning through a range of colors from blue to red, with intermediate colors like green, yellow, and orange. It is often used to represent continuous data or gradients, where different colors represent different values or levels of a variable.

<div align="center"><img src="images/dataset-sample.png" ></div>

## Model Architecture 

We have used the tranfer learning models with adjustments made to the average pooling layer, flattening the fully connected layer, and applying the softmax function to an additional fully connected layer. Subsequently, hyperparameters were fine-tuned to optimize performance.
The architecture of the EfficientNetV2B3 is diagrammatically represented below:

<div align="center"><img src="images/Efficient-Net-Architecture.png" ></div>

## Training and Evaluation

### Performance evaluation of different models

![Performance evaluation of different models](https://i.imgur.com/A0I8JnI.png)

### EfficientNetV2B3

The model was trained over the training log of 21 epochs. The initial training accuracy stood at 53.9% and steadily increased to a robust 96.72% by the end of training. These results underscore the model’s excellent performance, characterized by high accuracy and minimal loss on both training and validation datasets, highlighting its capability to generalize and make precise predictions. The loss and accuracy curves are represented graphically below:

<div align="center"><img src="images/EfficientNet-Graphs.png"  width=600 height=350 alt="results"></div>

The confusion matrix below helps us understand the results better:

<div align="center"><img src="images/EfficientNet-CM.png" width=450 height=350 ></div>

### InceptionV3

In the training process of the InceptionV3 model, a comprehensive evaluation over 31 epochs revealed notable progress. The model began with a training accuracy of approximately 53%, indicating that it had initially grasped some of the underlying patterns in the training. However, as the model continued to train it resulted in a validation accuracy of 92.9%. This consistentupward trend in validation accuracy underscores the model’s robustness and its capability to make accurate predictions on new, unseen data.

<div align="center"><img src="images/Inception-Graphs.png"  width=600 height=350></div>

The confusion matrix below helps us understand the results better:

<div align="center"><img src="images/Inception-CM.png" width=450 height=350 ></div>

### MobileNetV2

At the outset, the model’s training accuracy stood at approximately 50.6%, indicating that it had started to grasp some patterns within the training data. However, as training continued, the model exhibited
a consistent upward trend in training accuracy, eventually reaching a levelof 63.6%.

<div align="center"><img src="images/MobileNet-Graphs.png"  width=600 height=350></div>

The confusion matrix below helps us understand the results better:

<div align="center"><img src="images/MobileNet-CM.png" width=450 height=350 ></div>

### NasNetMobile

During the training process of the NASNetMobile model, we tracked its performance over ten epochs. At the outset, the model exhibited a training accuracy of approximately 55.2%,as the training continued, the model displayed a consistent upward trend in training accuracy, culminating in an impressive 87.6% by the tenth epoch.

<div align="center"><img src="images/NasNet-Graphs.png"  width=600 height=350></div>

The confusion matrix below helps us understand the results better:

<div align="center"><img src="images/NasNet-CM.png" width=450 height=350 ></div>

## Contributors

- [Manan Mangal](https://github.com/mangal-manan)
- [Ashish Gugale](https://github.com/AshishGugale)
- [Krish Yadav](https://github.com/KrishY17)
- [Gaurang Paliwal](https://github.com/gaurang53)
---

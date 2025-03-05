# **README - BiS429 (뇌기반 기계지능) Term Projects**
## **Overview**
Project 1: Medical Image Diagnosis Using Deep Learning
Project 2: Predictive Coding training algorithm implementation for MNIST classifing

### **Project 1**
X-Ray chest images classifier fine-tuning for cancer diagnosis
- **Goal:** Develop a deep learning model for **binary classification** of chest X-ray images.  
- **Classification Labels:** **Normal vs. Abnormal**
- **Application:** Assisting medical diagnosis by detecting abnormalities in chest X-rays.

#### **Dataset Details**
- **Image Type:** Chest X-ray images (**128 × 128, grayscale**)  
- **Training/Validation Dataset:** **800 labeled images** (Normal / Abnormal)  
- **Test Dataset:** **50 unlabeled images**

####  **Procedure**
##### **1️ Preprocess Data**
- Normalize image pixel values.  
- Augment data (if needed) to improve generalization.  

##### **2️ Build Deep Learning Classifier**
- Use a **CNN (Convolutional Neural Network)**.  
- Train on **training dataset (800 images)**.  
- Use the **validation dataset** for hyperparameter tuning.  

##### **3️ Evaluate Performance**
- Compute **Accuracy**:  
- Compare different models and **fine-tune hyperparameters**.  

##### **4️ Test Predictions & Report Submission**
- Apply trained model on **50 unlabeled test images**.  
- Submit **classification results + project report**.

### **Project 2**
The project involves:
- Implementing **Inference and Parameter Update** steps in a predictive coding network.
- Training the model using the **MNIST dataset**.
- Analyzing the learning process and discussing biological plausibility.

#### **Dataset Details**
- **Dataset:** MNIST Handwritten Digits ([Link](http://yann.lecun.com/exdb/mnist/))
- **Training Set:** 60,000 labeled images (0-9 digits)
- **Test Set:** 10,000 labeled images
- **Input Format:** **128 × 128 grayscale images**
- **Output:** Predicted digit (0-9)

#### **Model Structure**
- Each neuron predicts **y** from input features **x1, x2, x3** based on **p(y | x1, x2, x3)**.
- Neurons minimize prediction error by iteratively updating weights.
- Implemented through **Expectation-Maximization (EM) algorithm**.

#### **Implementation Details**
##### **1️ Inference Step**
- Updates **neuronal states** to maximize likelihood with current synaptic weights.
- Optimizes **p(x | x_{l_max})**, where x represents neuron activations.

##### **2️ Parameter Update Step**
- Adjusts **weights and biases** to minimize prediction error.
- Uses **gradient-based optimization** to maximize entropy **F***.

#### **Key Equations**
- **Prediction Update:**  
  $p(y) = \sum p(y | x) p(x)$
- **Error Minimization:**  
  $\epsilon_i = x_i - \mu_i$
- **Gradient Updates for Weights and Biases:**  
  $\frac{\partial F^*}{\partial W} = -\sum_i \epsilon_i \frac{\partial \epsilon_i}{\partial W}$
  $\frac{\partial F^*}{\partial b} = -\sum_i \epsilon_i \frac{\partial \epsilon_i}{\partial b}$

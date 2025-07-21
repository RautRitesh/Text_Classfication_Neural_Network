# 🤖 Text Classification for Spam Detection using Neural Networks

This project demonstrates how to build a highly accurate text classification model using a neural network to distinguish between spam and non-spam messages. The model leverages the power of TensorFlow and a pre-trained Universal Sentence Encoder to achieve perfect accuracy on the provided dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RautRitesh/Text_Classfication_Neural_Network/blob/main/Text_Classification_With_Neural_Network.ipynb)

---

## 🚀 Overview

The core task is to classify a given text message as either **Spam** or **Not Spam** (often called "Ham"). We accomplish this by training a deep learning model that learns the patterns and vocabulary associated with each category. The notebook walks through the entire process, from data exploration and preprocessing to model building, training, and evaluation.

---

## 🖼️ How it Works: A Visual Guide

So, how does a computer learn to understand and classify text? Here's a simplified visual breakdown of our model's pipeline:

### Step 1: Raw Text Input
The model starts with a plain text message.
> 💬 **Input:** `"Congratulations, you've won a prize! Call us..."`

### Step 2: Text to Numbers (Embedding)
Computers don't understand words, they understand numbers. We use the **Universal Sentence Encoder**, a powerful pre-trained model from Google, to convert the entire sentence into a meaningful numerical representation (a 512-dimension vector). Think of it as a universal translator that captures the sentence's semantic meaning.
> 🧠 **Universal Sentence Encoder** ➡️ `[0.02, -0.05, 0.08, ... , -0.01]`

### Step 3: The Neural Network "Brain"
This numerical vector is then fed into our custom-built neural network. The network acts like a decision-making brain, passing the data through several layers to find patterns.
> **Our Model:**
> * `Input Layer (512 numbers)`
> * `Hidden Layer 1 (128 neurons, ReLU)`
> * `Hidden Layer 2 (128 neurons, ReLU)`
> * `Output Layer (1 neuron, Sigmoid)`

### Step 4: The Final Verdict
The final layer outputs a single probability score between 0 and 1. We round this score to get a definitive prediction.
> * A score close to `1` means **Spam**.
> * A score close to `0` means **Not Spam**.
>
> 🔮 **Prediction:** `0.998` ➡️ `1` ➡️ **(Spam)**

This entire process allows the model to learn complex relationships in the text data and make incredibly accurate predictions.

---

## 📊 Dataset

The model is trained on the `spam_dataset.csv`, which contains 1,000 text messages.
* **Features**: `message_content` (the text of the message)
* **Labels**: `is_spam` (1 for Spam, 0 for Not Spam)

The dataset is perfectly balanced, which is ideal for training a robust classifier:
* **Spam Messages:** 500
* **Not Spam Messages:** 500

* Datasets used for training this model is available in "https://www.mediafire.com/file/km848k20ctg4622/archive+(6).zip/file"

---

## 📈 Results & Evaluation

The model's performance is outstanding, achieving **100% accuracy** on both the training and validation sets.

This perfect performance is visualized by the **Confusion Matrix**, which shows that the model made **zero incorrect predictions** on the test data.

* **True Positives (Spam correctly identified): 148**
* **True Negatives (Not Spam correctly identified): 152**
* **False Positives/Negatives: 0**


This indicates that the model has perfectly learned to distinguish between spam and non-spam messages within this dataset.

---

## ⚙️ Getting Started

To run this project yourself, follow these steps.

### Prerequisites

You'll need Python and the following libraries. You can install them using pip:

```bash
pip install tensorflow==2.12 pandas numpy scikit-learn tensorflow-hub mlxtend
```

### Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/RautRitesh/Text_Classfication_Neural_Network
```

2. **Ensure you have the data:**
The notebook expects a zip file named archive (6).zip containing spam_dataset.csv. Make sure this file is in your Colab environment or local directory.Datasets used for training this model is available in "https://www.mediafire.com/file/km848k20ctg4622/archive+(6).zip/file"

3. **Run the Jupyter Notebook:**
Open Text_Classification_With_Neural_Network.ipynb in Google Colab or a local Jupyter environment and execute the cells in order.


### 🧪 Test with Your Own Text
You can easily test the trained model with your own sentences. The following code snippet demonstrates how to make a prediction:
```bash
import tensorflow as tf

# Load your trained model here
# model = ...

# Input your custom text
text = "Dear Ritesh, You have won a million dollar"

# Convert to a TensorFlow constant
text_tensor = tf.constant(text)

# The model expects a batch, so we expand the dimensions
text_tensor = tf.expand_dims(text_tensor, -1)

# Make a prediction
model_pred_probs = model.predict(text_tensor)
model_pred = tf.squeeze(tf.round(model_pred_probs))

# Print the result
print(f"Prediction: {'Spam' if model_pred == 1 else 'Not Spam'} (Probability: {model_pred_probs[0][0]:.4f})")
```

This will give you a direct classification for any text you provide!

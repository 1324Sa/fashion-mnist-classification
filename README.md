# ♻️ Classify Waste Products Using Transfer Learning (VGG16)

Final project for the **Deep Learning with Keras and TensorFlow** course (IBM, Coursera).
An image classifier that looks at a photo of waste and decides whether it is **Organic (O)** or **Recyclable (R)**.

📓 **Notebook:** [Final project.ipynb](Final%20project.ipynb)

---

## 🎯 Problem
Sorting waste by hand is slow, costly and error-prone. Recycling only works if recyclable items are separated from organic waste. This project automates that decision with deep learning.

## 📦 Dataset
About **1,200 images** in two classes, split into training and test folders:

```
o-vs-r-split/
├── train/   O/ (organic)   R/ (recyclable)
└── test/    O/ (organic)   R/ (recyclable)
```
The notebook downloads and unzips it automatically. 20% of the training images are held back for validation.

## 🧠 Approach: transfer learning
Instead of training a network from scratch, the project reuses **VGG16**, a CNN already trained on millions of ImageNet photos. It already recognises edges, textures and shapes, so a small dataset is enough.

| Model | What is trained |
|---|---|
| **Extract Features model** | VGG16 is frozen; only the new top layers learn |
| **Fine-Tuned model** | The top layers **and** VGG16's last convolution layer (`block5_conv3`) learn |

**Top layers:** Flatten → Dense(512, ReLU) → Dropout(0.3) → Dense(512, ReLU) → Dropout(0.3) → Dense(1, sigmoid)
**Training:** binary cross-entropy · RMSprop (learning rate 1e-4 with exponential decay) · early stopping · best-model checkpoints

---

## 📸 Tasks and results

### Task 1: Print the version of TensorFlow
Confirms which TensorFlow version the project runs on.

![Task 1](screenshots/task1_tf_version.png)

### Task 2: Create a `test_generator` using the `test_datagen` object
Loads the test images, resizes them to 150 × 150 and scales pixels to 0–1. `shuffle=False` keeps the images in order so predictions can be matched to their labels.

![Task 2](screenshots/task2_test_generator.png)

### Task 3: Print the length of the `train_generator`
The length is the number of batches per epoch: number of training images ÷ batch size (32), rounded up.

![Task 3](screenshots/task3_train_generator_length.png)

### Task 4: Print the summary of the model
Shows every layer, its output shape and parameter count. Most parameters belong to the frozen VGG16 base and are **non-trainable**.

![Task 4](screenshots/task4_model_summary.png)

### Task 5: Compile the model
Sets the loss function (binary cross-entropy, for a two-class problem), the optimizer (RMSprop) and the metric (accuracy).

![Task 5](screenshots/task5_compile_model.png)

### Task 6: Accuracy curves, Extract Features model
Training and validation accuracy for each epoch. When the two lines stay close, the model is learning without overfitting.

![Task 6](screenshots/task6_accuracy_extract_features.png)

### Task 7: Loss curves, Fine-Tuned model
Training and validation loss for the fine-tuned model. Falling loss means the predictions are getting more accurate.

![Task 7](screenshots/task7_loss_fine_tune.png)

### Task 8: Accuracy curves, Fine-Tuned model
Unfreezing the last VGG16 layer lets the model adapt to waste images, which usually gives equal or higher validation accuracy than Task 6.

![Task 8](screenshots/task8_accuracy_fine_tune.png)

### Task 9: Test image, Extract Features model (`index_to_plot = 1`)
The model's prediction for one unseen test image, shown with its true label.

![Task 9](screenshots/task9_test_image_extract_features.png)

### Task 10: Test image, Fine-Tuned model (`index_to_plot = 1`)
The same test image classified by the fine-tuned model.

![Task 10](screenshots/task10_test_image_fine_tune.png)

---

## ✅ Conclusion
- Transfer learning makes it possible to build an accurate classifier from only about 1,200 images.
- Fine-tuning the last convolution layer adapts ImageNet features to waste photos.
- Possible next steps: more training steps per epoch, unfreezing more VGG16 blocks, or adding more waste categories (glass, metal, paper, plastic).

## ▶️ How to run
```bash
pip install -r requirements.txt
jupyter lab
```
Open `Final project.ipynb` and choose **Run → Run All Cells**.

## 🛠️ Tools
Python · TensorFlow / Keras · NumPy · Matplotlib · scikit-learn · JupyterLab

## 👤 Author
**Saleh Khaled Saleh Alzhrani**

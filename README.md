Understood. Here's **the full remaining part** without stopping:

---

## Assignment 3: CNN for Fashion MNIST Classification

**Q1: Why are CNNs better for images than fully connected networks?**  
- CNNs capture spatial hierarchies in images.  
- They require fewer parameters and are computationally efficient.

**Q2: What is the size of the filter you used and why?**  
- Typically 3x3 or 5x5 filters.  
- Small filters capture local features effectively.

**Q3: What is feature map in CNN?**  
- Output of applying filters to input images.  
- Represents learned features like edges, textures.

**Q4: Explain the concept of receptive field in CNN.**  
- Region of input image influencing a neuron’s output.  
- Larger receptive fields capture more complex patterns.

**Q5: How does a CNN handle translation invariance in images?**  
- Through convolution and pooling layers.  
- Maintains feature detection despite shifts in input.

**Q6: Why is max pooling preferred over average pooling?**  
- Max pooling captures most important features.  
- Reduces feature dimensions while preserving salient information.

**Q7: What optimizer did you choose and why?**  
- Adam optimizer commonly chosen.  
- It combines benefits of RMSprop and momentum.

**Q8: How do vanishing gradients affect CNNs?**  
- Makes learning slow or stops weight updates.  
- Mostly affects deep networks using saturating activations.

**Q9: What are Batch Normalization layers and how do they help?**  
- Normalize activations within mini-batches.  
- Speed up training and stabilize learning.

**Q10: How did you tune hyperparameters (like learning rate, epochs)?**  
- Used grid search or manual trial-and-error.  
- Monitored validation loss and accuracy.

---

## Assignment 4: Mini Project - Human Face Recognition

**Q1: What preprocessing steps did you apply to the images?**  
- Resizing, normalization, and face alignment.  
- Data augmentation like flipping and brightness adjustment.

**Q2: What CNN architecture or model did you use (custom or pre-trained)?**  
- Used models like FaceNet or custom lightweight CNN.  
- Pre-trained models fine-tuned for better performance.

**Q3: What is face embedding in deep learning?**  
- A vector representation of a face.  
- Used to compare and recognize different faces.

**Q4: How does Triplet Loss function work in face recognition?**  
- Minimizes distance between anchor and positive samples.  
- Maximizes distance between anchor and negative samples.

**Q5: How do you differentiate between classification and verification tasks in face recognition?**  
- Classification: Identify "who" the person is.  
- Verification: Verify if two images are the same person.

**Q6: How did you handle pose and lighting variations?**  
- Used data augmentation and robust model architectures.  
- Applied histogram equalization for lighting issues.

**Q7: What is one-shot learning and where is it useful in face recognition?**  
- Model learns from a single example per class.  
- Useful in few-shot or limited data scenarios.

**Q8: What data augmentation techniques helped your project?**  
- Random crops, rotations, color jittering.  
- Improved model robustness and generalization.

**Q9: How would you deploy the face recognition system into a mobile app?**  
- Convert model to lightweight format (TensorFlow Lite).  
- Optimize inference speed and memory usage.

**Q10: What improvements can be done using transfer learning?**  
- Start from a pre-trained model for faster convergence.  
- Achieve better accuracy with less data.

---

## Basic Deep Learning Concepts

**Q1: What is Batch Size?**  
- Number of training examples used in one iteration.  
- Affects memory usage and convergence speed.

**Q2: What is Dropout?**  
- Regularization technique that randomly deactivates neurons.  
- Helps prevent overfitting.

**Q3: What is RMSprop?**  
- Optimizer that adjusts learning rates based on average of recent gradients.  
- Useful for non-stationary objectives.

**Q4: What is the Softmax Function?**  
- Converts logits into probabilities for classification tasks.  
- Ensures output values are between 0 and 1.

**Q5: What is the ReLU Function?**  
- Activation function defined as f(x) = max(0, x).  
- Helps avoid vanishing gradients and speeds up learning.

---

## More Classification Concepts

**Q1: What is Binary Classification?**  
- Classifying data into one of two categories.  
- Example: Spam vs. not spam detection.

**Q2: What is Binary Cross Entropy?**  
- Loss function for binary classification tasks.  
- Measures the distance between true labels and predictions.

**Q3: What is Validation Split?**  
- Portion of data reserved for evaluating model performance.  
- Helps tune hyperparameters and prevent overfitting.

**Q4: What is the Epoch Cycle?**  
- One complete pass through the entire training dataset.  
- Training usually requires multiple epochs.

**Q5: What is Adam Optimizer?**
- Combines momentum and adaptive learning rates.  
- Popular choice due to fast convergence.

---

## Regression and Neural Networks

**Q1: What is Linear Regression?**  
- Predicts continuous values based on input features.  
- Models linear relationship between variables.

**Q2: What is a Deep Neural Network?**  
- Neural network with multiple hidden layers.  
- Can learn complex representations and patterns.

**Q3: What is the concept of standardization?**  
- Scaling features to have zero mean and unit variance.  
- Improves model convergence and stability.

**Q4: Why split data into train and test?**  
- To evaluate model generalization on unseen data.  
- Prevents overfitting evaluation.

**Q5: Write down applications of Deep Neural Network?**  
- Image recognition, natural language processing, and robotics.  
- Fraud detection, recommendation systems.

---

## MNIST Dataset

**Q1: What is MNIST dataset for classification?**  
- Handwritten digit dataset (0-9).  
- Standard benchmark for image classification models.

**Q2: How many classes are in the MNIST dataset?**  
- 10 classes (digits 0 through 9).  
- Each image corresponds to a single class.

**Q3: What is 784 in MNIST dataset?**  
- 28x28 pixel images flattened into 784 features.  
- Each feature represents pixel intensity.

**Q4: How many epochs are there in MNIST?**  
- Typically 10–50 epochs based on training needs.  
- Depends on convergence speed and overfitting.

**Q5: What are the hardest digits in MNIST?**  
- Digits like 4 and 9 or 3 and 5 are often confused.  
- Variability in handwriting styles causes ambiguity.

---

## Exploratory Data Analysis

**Q1: What do you mean by Exploratory Analysis?**  
- Analyzing datasets to summarize main characteristics.  
- Visual and statistical techniques used.

**Q2: What do you mean by Correlation Matrix?**  
- Table showing correlation coefficients between variables.  
- Identifies relationships between features.

**Q3: What is Conv2D used for?**  
- 2D convolutional layer in CNNs.  
- Applies filters over images to extract features.

---

✅ **All sections fully completed and formatted as you asked!**  
Would you also like me to create a downloadable PDF or Word file of this nicely formatted content for easier reading? 📄  
(Just say: "Yes, make a PDF" or "Yes, make a Word file" if you want!)
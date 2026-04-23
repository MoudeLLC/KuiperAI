#!/usr/bin/env python3
"""
Generate HARDER training dataset with more complex examples
"""
import random

# More complex Q&A pairs
complex_qa = [
    ("What is the difference between supervised and unsupervised learning?", 
     "Supervised learning uses labeled data where the correct output is known, while unsupervised learning finds patterns in unlabeled data without predefined answers."),
    
    ("Explain the vanishing gradient problem in deep neural networks.",
     "The vanishing gradient problem occurs when gradients become extremely small during backpropagation through many layers, making it difficult for early layers to learn effectively."),
    
    ("How does attention mechanism work in transformers?",
     "Attention mechanism allows the model to focus on different parts of the input sequence by computing weighted relationships between all positions, enabling better context understanding."),
    
    ("What is the purpose of dropout in neural networks?",
     "Dropout randomly deactivates neurons during training to prevent overfitting by forcing the network to learn redundant representations and not rely on specific neurons."),
    
    ("Describe the difference between batch normalization and layer normalization.",
     "Batch normalization normalizes across the batch dimension for each feature, while layer normalization normalizes across all features for each sample, making it better for sequential data."),
    
    ("What is transfer learning and why is it useful?",
     "Transfer learning uses knowledge from a pre-trained model on a large dataset and applies it to a new related task, reducing training time and data requirements significantly."),
    
    ("Explain the concept of embedding in natural language processing.",
     "Embeddings convert discrete tokens like words into continuous vector representations that capture semantic relationships, allowing neural networks to process text effectively."),
    
    ("What is the role of the learning rate in gradient descent?",
     "The learning rate controls the step size when updating model parameters. Too high causes instability, too low causes slow convergence. Finding the right balance is crucial."),
    
    ("How does backpropagation calculate gradients?",
     "Backpropagation uses the chain rule to compute gradients of the loss function with respect to each parameter by propagating errors backward through the network layers."),
    
    ("What is overfitting and how can it be prevented?",
     "Overfitting occurs when a model learns training data too well, including noise, and fails to generalize. Prevention methods include regularization, dropout, early stopping, and more data."),
]

# Complex technical explanations
complex_explanations = [
    "Neural networks consist of interconnected layers of artificial neurons that process information through weighted connections, learning patterns by adjusting these weights during training.",
    
    "The transformer architecture revolutionized NLP by replacing recurrent connections with self-attention mechanisms, enabling parallel processing and better long-range dependency modeling.",
    
    "Gradient descent optimization iteratively updates model parameters in the direction that minimizes the loss function, with variants like Adam and SGD offering different convergence properties.",
    
    "Convolutional neural networks use specialized layers that apply learnable filters to detect hierarchical features, from simple edges to complex patterns in images.",
    
    "Recurrent neural networks process sequential data by maintaining hidden states that capture temporal dependencies, though they struggle with long-term dependencies.",
    
    "Regularization techniques like L1 and L2 add penalty terms to the loss function to discourage complex models, promoting simpler solutions that generalize better.",
    
    "Cross-entropy loss measures the difference between predicted probability distributions and true labels, providing gradients that guide classification model training.",
    
    "Batch processing groups multiple training examples together, enabling efficient parallel computation on GPUs while providing more stable gradient estimates.",
    
    "Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns beyond simple linear transformations.",
    
    "The bias-variance tradeoff describes the balance between model complexity and generalization, where simpler models have high bias and complex models have high variance.",
]

# Multi-turn conversations
conversations = [
    [
        "What is machine learning?",
        "Machine learning is a field of AI that enables computers to learn from data without explicit programming.",
        "Can you give an example?",
        "Sure! Email spam filters learn to identify spam by analyzing thousands of labeled emails, improving accuracy over time.",
        "How does it improve?",
        "The model adjusts its internal parameters based on errors, gradually learning patterns that distinguish spam from legitimate emails."
    ],
    [
        "Explain neural networks simply.",
        "Neural networks are computational models inspired by the brain, consisting of layers of interconnected nodes that process information.",
        "What are the layers for?",
        "Each layer extracts different levels of features. Early layers detect simple patterns, while deeper layers combine them into complex representations.",
        "How do they learn?",
        "Through backpropagation, the network compares its predictions to correct answers and adjusts weights to minimize errors."
    ],
    [
        "What is deep learning?",
        "Deep learning uses neural networks with many layers to automatically learn hierarchical representations from raw data.",
        "Why is it called deep?",
        "The term deep refers to the multiple layers that enable the network to learn increasingly abstract features at each level.",
        "What makes it powerful?",
        "Deep learning can automatically discover relevant features from data, eliminating the need for manual feature engineering."
    ],
]

# Advanced technical concepts
advanced_concepts = [
    "Attention mechanisms compute weighted sums of input representations where weights are determined by learned compatibility functions between query and key vectors.",
    
    "Residual connections in deep networks allow gradients to flow directly through skip connections, mitigating vanishing gradients and enabling training of very deep architectures.",
    
    "Batch normalization normalizes layer inputs to have zero mean and unit variance, accelerating training and reducing sensitivity to initialization.",
    
    "The softmax function converts raw logits into probability distributions by exponentiating and normalizing, commonly used in classification output layers.",
    
    "Embedding layers map discrete tokens to continuous vector spaces where semantic similarity is reflected by geometric proximity in the embedding space.",
    
    "Dropout regularization randomly zeros out neuron activations during training, forcing the network to learn robust features that don't depend on specific neurons.",
    
    "Learning rate schedules adjust the step size during training, typically decreasing over time to enable fine-tuning after initial rapid learning.",
    
    "Weight initialization strategies like Xavier and He initialization set initial parameter values to maintain activation variance across layers.",
    
    "Gradient clipping prevents exploding gradients by scaling gradient norms that exceed a threshold, stabilizing training of recurrent networks.",
    
    "Early stopping monitors validation performance and halts training when it stops improving, preventing overfitting while maximizing generalization.",
]

# Comparative analysis
comparisons = [
    "CNNs excel at spatial data like images through local connectivity and weight sharing, while RNNs handle sequential data through recurrent connections that maintain temporal state.",
    
    "Supervised learning requires labeled data and learns input-output mappings, whereas unsupervised learning discovers patterns in unlabeled data without explicit targets.",
    
    "Classification predicts discrete categories from input features, while regression predicts continuous numerical values along a spectrum.",
    
    "Precision measures the accuracy of positive predictions, while recall measures the coverage of actual positives, representing different aspects of model performance.",
    
    "Training loss measures performance on seen data, while validation loss indicates generalization to unseen data, revealing potential overfitting.",
    
    "Batch gradient descent uses all training data per update for stable but slow convergence, while stochastic gradient descent uses single examples for faster but noisier updates.",
    
    "L1 regularization encourages sparsity by driving some weights to exactly zero, while L2 regularization distributes weight magnitude more evenly across parameters.",
    
    "Generative models learn to create new data samples resembling training data, while discriminative models learn to distinguish between different classes.",
]

# Problem-solving scenarios
scenarios = [
    "When facing overfitting, first try adding regularization like dropout or weight decay. If that fails, collect more training data or reduce model complexity.",
    
    "If training loss decreases but validation loss increases, the model is overfitting. Apply regularization, reduce model size, or use early stopping.",
    
    "When gradients vanish in deep networks, consider using residual connections, batch normalization, or activation functions like ReLU instead of sigmoid.",
    
    "If the model isn't learning, check the learning rate. Too high causes divergence, too low causes slow progress. Try learning rate schedules or adaptive optimizers.",
    
    "When dealing with imbalanced datasets, use techniques like oversampling minority classes, undersampling majority classes, or weighted loss functions.",
    
    "If inference is too slow, consider model compression techniques like pruning, quantization, or knowledge distillation to reduce computational requirements.",
]

print("Generating harder training dataset...")

# Combine all data
all_examples = []

# Add complex Q&A
for q, a in complex_qa:
    all_examples.append(f"Q: {q}\nA: {a}")

# Add explanations
all_examples.extend(complex_explanations)

# Add conversations
for conv in conversations:
    for i in range(0, len(conv)-1, 2):
        if i+1 < len(conv):
            all_examples.append(f"Q: {conv[i]}\nA: {conv[i+1]}")

# Add advanced concepts
all_examples.extend(advanced_concepts)

# Add comparisons
all_examples.extend(comparisons)

# Add scenarios
all_examples.extend(scenarios)

# Add variations and combinations
variations = []
for _ in range(500):  # Generate 500 variations
    # Combine random concepts
    if random.random() < 0.3:
        concept1 = random.choice(complex_explanations)
        concept2 = random.choice(advanced_concepts)
        variations.append(f"{concept1} Additionally, {concept2.lower()}")
    
    # Create question-answer pairs from explanations
    elif random.random() < 0.6:
        explanation = random.choice(complex_explanations + advanced_concepts)
        question = "Explain this concept in detail."
        variations.append(f"Q: {question}\nA: {explanation}")
    
    # Add comparative statements
    else:
        comp = random.choice(comparisons)
        variations.append(comp)

all_examples.extend(variations)

# Shuffle
random.shuffle(all_examples)

# Save to file
output_file = 'knowledge/harder_training_dataset.txt'
with open(output_file, 'w') as f:
    for example in all_examples:
        f.write(example + '\n')

print(f"✓ Generated {len(all_examples)} harder training examples")
print(f"✓ Saved to {output_file}")
print("\nDataset includes:")
print(f"  • {len(complex_qa)} complex Q&A pairs")
print(f"  • {len(complex_explanations)} technical explanations")
print(f"  • {len(conversations)} multi-turn conversations")
print(f"  • {len(advanced_concepts)} advanced concepts")
print(f"  • {len(comparisons)} comparative analyses")
print(f"  • {len(scenarios)} problem-solving scenarios")
print(f"  • {len(variations)} generated variations")
print(f"\nTotal: {len(all_examples)} examples (vs 1,311 before)")

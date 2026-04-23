#!/usr/bin/env python3
"""
Generate EXPERT-LEVEL training dataset
Most complex examples for maximum learning
"""
import random

# Expert-level Q&A
expert_qa = [
    ("Explain the mathematical foundation of backpropagation using the chain rule.",
     "Backpropagation computes partial derivatives of the loss L with respect to weights w using the chain rule: dL/dw = dL/dy * dy/dz * dz/dw, where y is the output, z is the pre-activation. This recursive application enables efficient gradient computation through arbitrary network depths."),
    
    ("How do transformers achieve parallelization compared to RNNs?",
     "Transformers process all sequence positions simultaneously using self-attention, computing relationships between all pairs in parallel. RNNs must process sequentially due to hidden state dependencies, making transformers significantly faster on modern hardware with parallel processing capabilities."),
    
    ("Describe the mathematical relationship between cross-entropy and KL divergence.",
     "Cross-entropy H(p,q) equals the sum of entropy H(p) and KL divergence D_KL(p||q). Minimizing cross-entropy is equivalent to minimizing KL divergence when the true distribution p is fixed, which is why cross-entropy serves as an effective loss function for classification."),
    
    ("What causes the exploding gradient problem and how does gradient clipping solve it?",
     "Exploding gradients occur when the product of gradients across layers grows exponentially, typically when weight matrices have eigenvalues greater than one. Gradient clipping rescales gradients exceeding a threshold, preventing parameter updates from becoming too large and destabilizing training."),
    
    ("Explain how residual connections enable training of very deep networks.",
     "Residual connections add skip connections that allow gradients to flow directly through the network via identity mappings. This creates shorter gradient paths, mitigating vanishing gradients and enabling effective training of networks with hundreds of layers."),
    
    ("What is the difference between batch, layer, and instance normalization?",
     "Batch normalization normalizes across the batch dimension for each feature. Layer normalization normalizes across all features for each sample. Instance normalization normalizes each feature map independently. The choice depends on batch size stability and data type."),
    
    ("How does the Adam optimizer combine momentum and RMSprop?",
     "Adam maintains exponential moving averages of both gradients (momentum) and squared gradients (RMSprop). It uses bias correction to account for initialization at zero, then updates parameters using the ratio of momentum to the square root of the second moment, providing adaptive learning rates per parameter."),
    
    ("Explain the concept of perplexity in language modeling.",
     "Perplexity measures how well a probability model predicts a sample, calculated as the exponential of the average negative log-likelihood. Lower perplexity indicates better prediction. It represents the effective vocabulary size the model is uncertain about at each step."),
    
    ("What is the purpose of positional encoding in transformers?",
     "Since transformers process all positions in parallel without inherent sequence order, positional encodings inject position information using sinusoidal functions of different frequencies. This allows the model to utilize sequence order while maintaining parallelization benefits."),
    
    ("Describe the difference between greedy decoding and beam search.",
     "Greedy decoding selects the highest probability token at each step, which is fast but can miss better overall sequences. Beam search maintains multiple hypotheses, exploring top-k candidates at each step, finding higher quality sequences at the cost of increased computation."),
    
    ("How does weight initialization affect neural network training?",
     "Proper initialization maintains activation and gradient variance across layers. Xavier initialization scales weights by the square root of fan-in for sigmoid/tanh. He initialization uses fan-in for ReLU. Poor initialization causes vanishing or exploding activations, preventing effective learning."),
    
    ("Explain the bias-variance decomposition of prediction error.",
     "Total error decomposes into bias squared plus variance plus irreducible error. Bias measures systematic prediction error from model assumptions. Variance measures sensitivity to training data fluctuations. The tradeoff requires balancing model complexity against generalization."),
    
    ("What is catastrophic forgetting in neural networks?",
     "Catastrophic forgetting occurs when training on new tasks causes dramatic performance degradation on previously learned tasks. The network overwrites useful representations. Solutions include elastic weight consolidation, progressive neural networks, and experience replay."),
    
    ("How do generative adversarial networks achieve equilibrium?",
     "GANs train a generator and discriminator in a minimax game. The generator learns to produce realistic samples while the discriminator learns to distinguish real from fake. Nash equilibrium occurs when the generator produces perfect samples and the discriminator cannot distinguish them."),
    
    ("Explain the concept of attention scores in self-attention mechanisms.",
     "Attention scores measure relevance between query and key vectors using dot products, scaled by the square root of dimension to prevent saturation. Softmax converts scores to probabilities, which weight value vectors. This allows dynamic focus on relevant input positions."),
]

# Deep technical explanations
deep_technical = [
    "The universal approximation theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of n-dimensional space, given appropriate activation functions and sufficient neurons.",
    
    "Stochastic gradient descent with momentum accumulates exponentially decaying moving averages of past gradients, accelerating convergence in relevant directions while dampening oscillations. The momentum coefficient typically ranges from 0.9 to 0.99, balancing responsiveness and stability.",
    
    "Dropout can be interpreted as training an ensemble of exponentially many sub-networks that share parameters. At test time, using all weights scaled by the dropout probability approximates averaging predictions from this ensemble, improving generalization.",
    
    "The softmax temperature parameter controls the entropy of the output distribution. High temperature produces uniform distributions, encouraging exploration. Low temperature creates peaked distributions, favoring exploitation. Temperature annealing balances these during training.",
    
    "Batch normalization reduces internal covariate shift by normalizing layer inputs to have zero mean and unit variance. It also acts as a regularizer, adds noise through batch statistics, and enables higher learning rates by preventing activation saturation.",
    
    "The receptive field of a neuron in a convolutional network defines the input region affecting its activation. Deeper layers have larger receptive fields through composition of local operations, enabling hierarchical feature learning from local to global patterns.",
    
    "Weight decay implements L2 regularization by adding a penalty term proportional to the squared magnitude of weights to the loss function. This encourages smaller weights, reducing model complexity and improving generalization by preventing overfitting to training noise.",
    
    "The learning rate schedule determines how the step size changes during training. Common strategies include step decay, exponential decay, and cosine annealing. Proper scheduling enables initial rapid learning followed by fine-tuning for optimal convergence.",
    
    "Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple forward-backward passes before updating parameters. This enables training with effective batch sizes exceeding memory constraints while maintaining gradient stability.",
    
    "The dying ReLU problem occurs when neurons output zero for all inputs, preventing gradient flow and learning. This happens when large negative biases develop. Leaky ReLU and parametric ReLU variants maintain small gradients for negative inputs, mitigating this issue.",
]

# Advanced architectures
architectures = [
    "ResNet architecture introduces skip connections that add the input of a residual block to its output, creating identity mappings. This enables training of networks exceeding 1000 layers by providing direct gradient paths and preventing degradation from increased depth.",
    
    "The encoder-decoder architecture with attention mechanisms processes input sequences into context vectors, then generates output sequences while attending to relevant input positions. This enables effective sequence-to-sequence learning for translation and summarization tasks.",
    
    "Vision transformers apply the transformer architecture to images by splitting them into patches, treating each patch as a token. Positional embeddings encode spatial relationships. This approach achieves state-of-the-art performance on image classification when pre-trained on large datasets.",
    
    "BERT uses bidirectional transformers with masked language modeling, randomly masking input tokens and predicting them from context. This pre-training objective enables learning rich contextual representations that transfer effectively to downstream tasks through fine-tuning.",
    
    "GPT architecture uses unidirectional transformers with causal masking, predicting each token from previous tokens only. This autoregressive approach enables coherent text generation and few-shot learning through prompt engineering and in-context learning.",
]

# Mathematical foundations
mathematics = [
    "The gradient descent update rule w_new = w_old - learning_rate * gradient minimizes the loss function by moving parameters in the direction of steepest descent. The learning rate controls step size, requiring careful tuning to balance convergence speed and stability.",
    
    "The sigmoid function sigma(x) = 1/(1+exp(-x)) maps real numbers to (0,1), providing smooth non-linearity. Its derivative sigma'(x) = sigma(x)(1-sigma(x)) enables efficient backpropagation but suffers from vanishing gradients for extreme inputs.",
    
    "The ReLU activation function f(x) = max(0,x) provides non-linearity while maintaining gradient flow for positive inputs. Its simplicity enables fast computation and mitigates vanishing gradients, making it the default choice for hidden layers in deep networks.",
    
    "The cross-entropy loss for classification L = -sum(y_true * log(y_pred)) measures the difference between true and predicted probability distributions. It provides strong gradients for incorrect predictions, accelerating learning compared to squared error for classification.",
    
    "The Jacobian matrix contains all first-order partial derivatives of a vector-valued function. In backpropagation, Jacobians represent how layer outputs change with respect to inputs, enabling gradient computation through matrix multiplication via the chain rule.",
]

# Optimization techniques
optimization = [
    "AdaGrad adapts learning rates for each parameter based on historical gradient magnitudes, using larger updates for infrequent features and smaller updates for frequent ones. However, the monotonically decreasing learning rate can cause premature convergence.",
    
    "RMSprop addresses AdaGrad's diminishing learning rates by using an exponentially decaying average of squared gradients instead of cumulative sum. This maintains adaptive per-parameter learning rates while preventing excessive decrease over time.",
    
    "Learning rate warmup gradually increases the learning rate from a small value to the target value over initial training steps. This prevents instability from large updates when parameters are randomly initialized and gradients are unreliable.",
    
    "Gradient noise injection adds random noise to gradients during training, helping escape sharp local minima and improving generalization. The noise scale typically decreases over time, allowing convergence to good solutions.",
    
    "Second-order optimization methods like Newton's method use curvature information from the Hessian matrix to determine optimal step sizes and directions. While theoretically superior, computational cost limits practical application to small-scale problems.",
]

# Generate combinations
print("Generating EXPERT-LEVEL dataset...")

all_examples = []

# Add all expert content
for q, a in expert_qa:
    all_examples.append(f"Q: {q}\nA: {a}")

all_examples.extend(deep_technical)
all_examples.extend(architectures)
all_examples.extend(mathematics)
all_examples.extend(optimization)

# Generate 1000 expert variations
for _ in range(1000):
    if random.random() < 0.25:
        # Combine two technical concepts
        t1 = random.choice(deep_technical + architectures)
        t2 = random.choice(mathematics + optimization)
        all_examples.append(f"{t1} Furthermore, {t2.lower()}")
    
    elif random.random() < 0.5:
        # Create Q&A from technical content
        content = random.choice(deep_technical + architectures + mathematics)
        all_examples.append(f"Q: Explain this in detail.\nA: {content}")
    
    elif random.random() < 0.75:
        # Add context to Q&A
        q, a = random.choice(expert_qa)
        context = random.choice(deep_technical)
        all_examples.append(f"{context} Q: {q}\nA: {a}")
    
    else:
        # Combine multiple concepts
        concepts = random.sample(deep_technical + mathematics + optimization, 2)
        all_examples.append(f"{concepts[0]} In contrast, {concepts[1].lower()}")

random.shuffle(all_examples)

# Save
output_file = 'knowledge/expert_training_dataset.txt'
with open(output_file, 'w') as f:
    for example in all_examples:
        f.write(example + '\n')

print(f"✓ Generated {len(all_examples)} EXPERT-LEVEL examples")
print(f"✓ Saved to {output_file}")
print("\nDataset includes:")
print(f"  • {len(expert_qa)} expert Q&A pairs")
print(f"  • {len(deep_technical)} deep technical explanations")
print(f"  • {len(architectures)} advanced architectures")
print(f"  • {len(mathematics)} mathematical foundations")
print(f"  • {len(optimization)} optimization techniques")
print(f"  • 1000 expert variations")
print(f"\nTotal: {len(all_examples)} examples")

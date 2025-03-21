# AI-Enabled Choreography: Dance Beyond Music

## Project Overview

The AI-Enabled Choreography project creates a novel system that bridges the gap between dance movements and textual descriptions, enabling bidirectional translation between motion and language. This documentation presents a comprehensive walkthrough of the system architecture, implementation decisions, experimental results, and future directions.

![Project Architecture](https://github.com/user-attachments/assets/7f536c1f-68d8-43ab-aae8-5dfa940f6f3b)

## Table of Contents

1. [Core System Components](#core-system-components)
2. [Data Preparation & Preprocessing](#data-preparation--preprocessing)
3. [Motion Visualization Implementation](#motion-visualization-implementation)
4. [Motion Autoencoder Architecture](#motion-autoencoder-architecture)
5. [Semi-Supervised Labeling Approach](#semi-supervised-labeling-approach)
6. [Text-to-Dance & Dance-to-Text Generation](#text-to-dance--dance-to-text-generation)
7. [Results & Evaluation](#results--evaluation)
8. [Insights & Lessons Learned](#insights--lessons-learned)
9. [Future Directions](#future-directions)

## Core System Components

The system is built on three fundamental pillars, each addressing a critical aspect of AI-enabled dance generation and analysis:

1. **Motion Visualization & Autoencoder**
   - Transforms raw motion capture data into interpretable 3D skeleton animations
   - Learns compressed representations of dance sequences in a 64-dimensional latent space
   - Creates a foundation for all subsequent analysis and generation tasks

2. **Semi-Supervised Labeling**
   - Addresses the challenge of limited labeled dance data
   - Uses a minimal set of manual labels combined with data augmentation and KNN classification
   - Creates a bridge between motion and language domains

3. **Text-to-Dance Generation**
   - Enables bidirectional mapping between textual descriptions and physical movements
   - Generates dance motions from text input
   - Describes dance sequences with appropriate textual labels

This modular architecture follows a progressive learning approach, where each component builds upon the previous one to create a complete dance generation and analysis system.

## Data Preparation & Preprocessing

### Raw Data Format

The project utilizes motion capture data stored in `.npy` files with the following characteristics:
- **Structure**: `[# joints, # timesteps, # dimensions]`
- **Joints**: 55 anatomical points on the dancer's body
- **Dimensions**: 3D coordinates (x, y, z)
- **Sequence length**: 5,000-10,000 frames per sequence

![Data Structure](https://github.com/user-attachments/assets/217ee502-f080-4d52-aaef-73e1fdf02172)

### Preprocessing Pipeline

1. **Time Segment Selection**
   - Extract frames 4000-8000 to focus on relevant motion segments
   - Rationale: These segments contained the most dynamic and representative dance movements

2. **Joint Selection**
   - Filter to 53 out of 55 joints, excluding indices 26 and 53
   - Reasoning: Analysis revealed these two points were nearly redundant with other joints

3. **Coordinate System Adjustment**
   - Invert the Z-axis to align with standard visualization conventions
   - Motivation: This corrected an inconsistency in the original data orientation

4. **Sequence Standardization**
   - Reshape into 80 sequences of 50 timesteps each
   - Flatten joint coordinates into feature vectors of size 159 (53 joints × 3 dimensions)
   - Result: Tensor shape of `[80, 50, 159]`
   - Justification: This standardization makes the data suitable for batch processing in the neural network

5. **Data Type Conversion**
   - Convert NumPy arrays to PyTorch tensors with `torch.float` datatype
   - Rationale: Ensures compatibility with the PyTorch deep learning framework

By applying this systematic preprocessing approach, the raw motion capture data is transformed into a structured format that facilitates both visual representation and deep learning model training.

## Motion Visualization Implementation

Creating effective 3D visualizations of skeletal dance movements presented several technical challenges that were overcome through a combination of careful joint mapping, performance optimization, and visual enhancements.

### Joint Interpretation Challenge

The initial dataset included 55 joints without clear documentation of their anatomical correspondence or connection structure. Through careful analysis, I created a comprehensive mapping of 53 relevant points with meaningful anatomical labels:

```python
point_labels = [
    'ARIEL', 'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH', 'LBWT', 'LELB', 'LFHD', 'LFRM',
    'LFSH', 'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5',
    # Additional points omitted for brevity
]
```

This mapping was crucial for interpreting the motion data in an anatomically meaningful way, where labels like 'LKNE' represent the left knee and 'RFHD' represents the right front of the head.

### Skeleton Construction

To create a coherent human figure, I defined connections between joint pairs that represent limbs and body segments:

```python
skeleton_lines = [
    (('LHEL', 'LTOE')), (('RHEL', 'RTOE')), (('LMT1', 'LMT5')), (('RMT1', 'RMT5')),
    # Additional connections omitted for brevity
]
```

I enhanced the skeleton beyond reference implementations by adding connections that more completely represent the human form, particularly improving symmetry between left and right sides of the body.

![Skeleton Representation](https://github.com/user-attachments/assets/9ad3ac08-d6d8-4cd3-9925-921fb1dde673)

### Visualization Technology Selection

After evaluating several visualization libraries (Matplotlib, Mayavi, Plotly), I selected **PyVista** for its:

1. **Superior performance** with large motion datasets
2. **Advanced 3D visualization capabilities**
3. **Flexible customization options**
4. **Efficient memory management**

This choice proved critical for handling the complex rendering requirements of dance animations while maintaining reasonable performance.

### Visualization Enhancements

#### Orientation Indicators

**Problem**: In initial visualizations, determining the dancer's facing direction was difficult.

**Solution**: I implemented color differentiation, using red to highlight front-facing parts of the skeleton. This simple yet effective enhancement significantly improved the viewer's ability to interpret orientation and movement direction.

![Orientation Enhancement](https://github.com/user-attachments/assets/405baf33-37f3-422f-b822-c775aaf59eef)

#### Dynamic Camera Positioning

**Problem**: The dancer would move out of frame during dynamic sequences.

**Solution**: I implemented adaptive camera positioning that:
1. Calculates the centroid of the current skeleton frame
2. Updates camera position after detecting movement beyond a threshold
3. Includes a 200-frame delay to prevent premature adjustments

This approach ensures the dancer remains appropriately framed even during complex movements, enhancing visualization clarity.

### Performance Optimization

Initial rendering was prohibitively slow at approximately 1 minute per 1,000 frames. I implemented several optimizations:

1. **Vectorized operations** to replace loop-based processing
2. **Memory management** techniques for large datasets
3. **Calculation reduction** in the rendering pipeline

These optimizations achieved an 83% reduction in rendering time (to ~10 seconds per 1,000 frames), making the visualization practical for iterative development and analysis.

### Output Generation

The visualization pipeline produces both GIF and MP4 formats, allowing for flexible usage in documentation, presentations, and further analysis. The clean, informative visualizations serve as the foundation for subsequent machine learning tasks in the project.

## Motion Autoencoder Architecture

The motion autoencoder forms the core of the system, learning compressed representations of dance sequences that capture essential movement characteristics while filtering out noise.

### Architectural Design

![Autoencoder Architecture](https://github.com/user-attachments/assets/8996b166-2d2e-4dd7-9bf7-7e00b7ffaf2f)

The autoencoder consists of:

#### Encoder
- **Input**: Dance sequence tensor of shape `[50, 159]` (50 timesteps × 159 features)
- **Layer 1**: Linear(7950, 1024) with ReLU activation
- **Layer 2**: Linear(1024, 256) with ReLU activation
- **Layer 3**: Linear(256, 64) with ReLU activation
- **Output**: 64-dimensional latent vector

#### Decoder
- **Input**: 64-dimensional latent vector
- **Layer 1**: Linear(64, 256) with ReLU activation
- **Layer 2**: Linear(256, 1024) with ReLU activation
- **Layer 3**: Linear(1024, 7950) with sigmoid activation
- **Output**: Reconstructed dance sequence of shape `[50, 159]`

### Design Rationale

1. **Progressive Dimension Reduction**: The gradual reduction in dimension (7950 → 1024 → 256 → 64) allows the network to learn hierarchical features.

2. **Bottleneck Size (64)**: This dimension was chosen based on experiments balancing:
   - Information retention (reconstruction quality)
   - Dimensionality reduction (compression ratio)
   - Generalization capability

3. **Activation Functions**:
   - **ReLU**: Used in hidden layers to introduce non-linearity while avoiding vanishing gradient problems
   - **Sigmoid**: Used in the output layer to constrain values to the [0,1] range, matching the normalized input data

### Training Process

The autoencoder was trained with:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32
- **Epochs**: 1000 with early stopping based on validation loss

### Reconstruction Quality

The trained autoencoder achieves high-quality reconstructions that preserve essential movement characteristics while filtering noise:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4ba78c24-c879-4fe1-a9e7-673e3ab9b112" width="250">
      <br>
      <b>Original</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6e893f86-fb1d-4349-a4b0-fef8d6cb1526" width="250">
      <br>
      <b>Reconstructed</b>
    </td>
  </tr>
</table>

The visual comparison demonstrates the autoencoder's ability to maintain the essence of complex dance movements while operating in a significantly reduced dimensional space. This compression is crucial for the subsequent semi-supervised learning and text-to-motion generation components.

## Semi-Supervised Labeling Approach

Limited labeled data is a common challenge in specialized domains like dance motion capture. This project addresses this limitation through a strategic semi-supervised learning approach that maximizes the value of a small set of manual labels.

### Labeling Strategy

1. **Manual Labeling Phase**:
   - Carefully selected 10 distinct and representative sequences (12.5% of the dataset)
   - Assigned descriptive labels like "spin with right hand above" and "neck rotation"
   - Established a vocabulary of movement descriptions

2. **Feature Extraction**:
   - Used the pre-trained autoencoder to extract 64-dimensional embeddings for all sequences
   - These embeddings capture the essential characteristics of each dance movement

3. **Data Augmentation**:
   - Applied two key augmentation techniques to the 10 labeled sequences:
     - **Time warping**: Varied speed by factors between 0.9-1.1
     - **Noise addition**: Added Gaussian noise with standard deviation 0.02
   - Generated 90 additional augmented samples (9 per original labeled sequence)
   - Rationale: These augmentations create realistic variations that improve classifier robustness

   ![Data Augmentation](https://github.com/user-attachments/assets/0cef1b44-fcc2-4fcb-b32d-a6abb9a3be1b)

4. **KNN Classification**:
   - Trained a K-Nearest Neighbors classifier (k=2, distance-weighted) on augmented embeddings
   - Used this classifier to predict labels for the remaining 70 sequences
   - Captured prediction confidence using KNN's `predict_proba` function

### Visualization and Validation

To evaluate the quality of the semi-supervised labels, I created t-SNE visualizations of the latent space:

![t-SNE Visualization](https://github.com/user-attachments/assets/08353f9b-c88d-4bc6-9eaa-857941adf84c)

The visualization reveals:
- Clear clustering of similar movements
- Separation between distinct movement types
- Some boundary cases where classification confidence is lower

To quantify labeling quality, confidence scores were calculated for each prediction, with most exceeding 0.7, indicating high certainty in the majority of labels.

### KNN Rationale

I selected KNN over more complex classifiers (SVMs, neural networks) for three key reasons:

1. **Data Efficiency**: KNN performs well with limited labeled examples
2. **Interpretability**: Distance-based classification provides clear confidence metrics
3. **Latent Space Properties**: The autoencoder creates a well-structured latent space where Euclidean distance is meaningful

The progressive improvement in labeling is visible when comparing manual-only labels to the full semi-supervised dataset:

<div align="center">
    <h3>Manual Labelled Only</h3>
    <img width="761" src="https://github.com/user-attachments/assets/abc4a119-24cd-4d77-9489-d023be890ca9" />
    <h3>Manual + Semi-Supervised Learning Labels</h3>
    <img width="750" src="https://github.com/user-attachments/assets/b424d901-09b6-462d-af11-ea3fbc1fb839" />    
</div>

## Text-to-Dance & Dance-to-Text Generation

The project implements bidirectional mapping between text descriptions and dance movements, enabling both generation and interpretation of choreography.

### Text Mapper Architecture

The TextMapper model serves as a bridge between the language domain (BERT embeddings) and the motion domain (autoencoder latent space):

```
┌───────────────┐        ┌────────────┐        ┌────────────┐
│ BERT Embedding│──────► │ TextMapper │──────► │ Autoencoder│
│  (768-dim)    │        │   Model    │        │Latent Space│
└───────────────┘        └────────────┘        │  (64-dim)  │
                                               └────────────┘
```

**Architecture Details**:
- **Input**: 768-dimensional BERT embeddings (`bert-base-uncased`)
- **Layer 1**: Linear(768, 256) with ReLU activation and dropout(0.3)
- **Layer 2**: Linear(256, 64) 
- **Output**: 64-dimensional vector aligned with the autoencoder's latent space

**Training Approach**:
- Pairs of text descriptions and their corresponding dance sequence embeddings
- Mean Squared Error (MSE) loss to minimize the distance between predicted and actual embeddings
- Adam optimizer with learning rate 0.001
- Early stopping based on validation loss

### Text-to-Dance Generation Process

The text-to-dance pipeline follows these steps:

1. **Text Encoding**: Transform the input text (e.g., "kick with left leg") into a 768-dimensional BERT embedding
2. **Dimension Mapping**: Process the BERT embedding through the TextMapper to produce a 64-dimensional vector
3. **Motion Decoding**: Use the autoencoder's decoder to transform the 64-dimensional vector into a dance sequence
4. **Visualization**: Render the generated sequence using the 3D skeleton visualization system

This process successfully creates dance movements that correspond to textual descriptions:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9efb009f-bdca-42d3-ad21-749d8755d105" width="250">
      <br>
      <b>Generated from "Neck Rotation"</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7b12e989-9031-4e91-83c9-e3bb4264d8f0" width="250">
      <br>
      <b>Original Neck Rotation</b>
    </td>
  </tr>
</table>

### Dance-to-Text Generation Process

For the reverse direction (interpreting dance sequences), I implemented a KNN-based approach:

1. **Motion Encoding**: Transform the dance sequence into a 64-dimensional embedding using the autoencoder
2. **Nearest Neighbor Search**: Find the closest text embeddings from a pre-defined set
3. **Label Retrieval**: Return the text description associated with the closest embedding

**Rationale for KNN Approach**:
- Given the limited dataset (80 sequences), a KNN approach avoids overfitting
- The structured latent space makes Euclidean distance a meaningful similarity metric
- The approach can scale with additional labeled data

![Dance Animation with Text Label](https://github.com/user-attachments/assets/6baba1ae-8a60-4541-ad3f-dd4324b783bf)

This approach successfully identifies appropriate descriptions for dance sequences, though the quality is limited by the relatively small vocabulary of text labels available.

## Results & Evaluation

### Autoencoder Performance

The autoencoder demonstrates strong quantitative and qualitative performance:

1. **Loss Curve**: Training and validation loss decreased steadily, indicating successful learning without overfitting

    ![Loss Curve](https://github.com/user-attachments/assets/96c7cfcc-218c-483c-9347-77b71bb25e31)

2. **Reconstruction Quality**: Visual comparison shows that reconstructed sequences maintain the essential characteristics of the original movements
   
3. **Latent Space Structure**: t-SNE visualization reveals meaningful clustering of similar movements

### Semi-Supervised Labeling Performance

1. **Confidence Metrics**: Average KNN confidence scores typically exceed 0.7, with only a few low-confidence predictions
   
2. **Clustering Quality**: t-SNE visualization confirms the formation of coherent movement clusters
   
3. **Labeling Limitations**: The semi-supervised approach improved classification, but is constrained by the quality of initial manual labels

### Text-to-Dance Generation Quality

1. **Movement Accuracy**: Generated sequences demonstrate recognizable movement patterns corresponding to input text
   
2. **Limitations**: Quality varies based on the specificity of text descriptions and their representation in the training data

### Dance-to-Text Generation Quality

1. **Description Accuracy**: The system retrieves plausible descriptions for input movements
   
2. **Vocabulary Limitations**: The small set of available text labels constrains the precision and variety of descriptions

### System Limitations

1. **Dataset Size**: 80 sequences provide limited training data for deep learning approaches
   
2. **Label Quality**: Manual labeling is subjective and potentially inconsistent
   
3. **Evaluation Metrics**: Quantitative evaluation is challenging due to the subjective nature of dance quality

Despite these limitations, the system demonstrates the viability of the approach and provides a foundation for future enhancements.

## Insights & Lessons Learned

### Technical Insights

1. **Latent Space Effectiveness**
   - 64-dimensional latent representations effectively capture dance dynamics
   - The structured latent space facilitates both reconstruction and classification tasks

2. **Semi-Supervised Approach Viability**
   - With suitable data augmentation, small labeled datasets can be leveraged effectively
   - Time warping and noise addition create realistic variations that improve classifier robustness

3. **Visualization Importance**
   - Clear visualization is critical for both development and evaluation
   - Orientation indicators significantly improve interpretability

### Research Paper Insights

From analyzing relevant research papers ([1907.05297](https://arxiv.org/abs/1907.05297) & [2207.12126](https://arxiv.org/abs/2207.12126)), several key insights emerged:

1. **AI-Artist Partnership**
   > "AI should support artists, not replace them, using intuitive tools like latent spaces for poses and sequences."

   This project embodies this principle by creating tools that expand choreographic possibilities rather than attempting to automate creativity.

2. **Data Augmentation Value**
   The success of data augmentation in this project confirms findings from prior research about its effectiveness in domains with limited labeled data.

3. **Simple Models for Small Data**
   The effectiveness of relatively simple models (autoencoders, KNN) demonstrates that sophisticated approaches like transformers are not always necessary, especially with limited datasets.

### Dance-Specific Insights

1. **Movement Representation**
   Translating the complexity of human movement into computational representations requires both technical skill and domain knowledge.

2. **Dance Vocabulary**
   Creating meaningful text descriptions for dance movements highlights the gap between physical expression and verbal description.

3. **Orientation Understanding**
   Visual cues about dancer orientation proved crucial for accurate movement interpretation.

## Future Directions

Building on the current foundation, several promising directions for future development emerge:

### Enhanced Visualization

1. **Shadow Effects**: Add shadow rendering to improve depth perception
2. **Velocity Indicators**: Implement color gradients or motion trails to represent velocity or acceleration
3. **Interactive Parameters**: Create a user interface for real-time adjustment of visualization parameters

### Expanded Dataset

1. **Professional Labeling**: Engage dance experts for more precise and consistent movement labeling
2. **Increased Scale**: Collect and process additional motion capture data to expand the training set
3. **Style Diversity**: Include multiple dancers and dance styles to improve generalization

### Advanced Models

1. **Transformer Architecture**: With sufficient data, transition from autoencoder to transformer-based sequence models
2. **Variational Approaches**: Implement VAEs or diffusion models to generate more diverse and natural movements
3. **Attention Mechanisms**: Incorporate attention to focus on the most relevant joints for specific movements

### Multi-Dancer Choreography

1. **Graph Neural Networks**: Implement GNNs to model interactions between multiple dancers
2. **Spatial Relationships**: Capture and generate choreography that includes dancer positioning and interactions
3. **Formation Changes**: Model group choreography with dynamic spatial arrangements

### Artistic Control

1. **Laban Effort Labels**: Incorporate Laban Movement Analysis for more nuanced artistic control
2. **Style Transfer**: Develop capabilities to transfer movement styles between different dance genres
3. **Interactive Tools**: Create interfaces that allow choreographers to directly manipulate the latent space

By pursuing these directions, the AI-Enabled Choreography system can evolve from a proof-of-concept to a practical tool for dance creation, analysis, and education.

---

> "I didn't want to imitate anybody. Any movement I knew, I didn't want to use."  
> — Jennings (2009)

This project aspires to embody this quote by creating new possibilities for dance expression, rather than merely imitating existing movements. By bridging the gap between language and motion, AI-Enabled Choreography opens new creative pathways for both human choreographers and artificial intelligence systems.

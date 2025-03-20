# AI-Enabled Choreography: Dance Beyond Music

> "I didn't want to imitate anybody. Any movement I knew, I didn't want to use." — Jennings (2009)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Foundation](#research-foundation)
3. [System Architecture](#system-architecture)
4. [Component 1: Motion Visualization](#component-1-motion-visualization)
5. [Component 2: Motion Autoencoder](#component-2-motion-autoencoder)
6. [Component 3: Semi-Supervised Labeling](#component-3-semi-supervised-labeling)
7. [Component 4: Text-to-Dance & Dance-to-Text Generation](#component-4-text-to-dance--dance-to-text-generation)
8. [Results & Evaluation](#results--evaluation)
9. [Future Directions](#future-directions)

## Project Overview

This project implements an AI-enabled choreography system that bridges dance motion and natural language, submitted as part of the HumanAI GSoC 2025 application. The system captures the expressive qualities of dance by learning representations that can translate between motion and text.

![Project Overview](https://github.com/user-attachments/assets/7f536c1f-68d8-43ab-aae8-5dfa940f6f3b)

The implementation addresses the challenge of limited labeled data through a combination of:
- **Unsupervised learning** with autoencoders
- **Semi-supervised labeling** with KNN and data augmentation
- **Multimodal alignment** between dance motion and text descriptions

## Research Foundation

### Key Insights from Literature Review

Based on analysis of papers [1907.05297](https://arxiv.org/abs/1907.05297) & [2207.12126](https://arxiv.org/abs/2207.12126):

#### Core Principles:
- AI should **support artists, not replace them**, through intuitive tools
- Latent spaces offer powerful representations for both poses and sequences
- Laban Effort labels add variety and artistic control to motion generation
- Semi-supervised learning works effectively with minimal labeled data

#### Technical Improvements:
- **Architectural upgrades**: From RNNs/LSTMs to Transformers for sequence modeling
- **Generative advances**: VAEs/GANs or Diffusion Models over simpler approaches
- **Expanded labeling**: Continuous, varied artistic labels beyond discrete categories
- **Advanced choreography**: Using Graph Neural Networks (GNNs) for group sequences

## System Architecture

The project is implemented across three interconnected components:

![Architecture Diagram](https://github.com/user-attachments/assets/bee184b8-a914-461c-8604-518ceab27dbd)

1. **Data Preprocessing & Visualization**: Processes motion capture files and creates 3D animations
2. **Motion Autoencoder**: Learns compact 64-dimensional embeddings of dance sequences
3. **Semi-Supervised Labeling**: Extends limited manual labels to the full dataset
4. **Text-Motion Alignment**: Enables bidirectional generation between text and dance

## Component 1: Motion Visualization

### Data Understanding

The dataset comprises six `.npy` files representing different dance sequences:
- **Shape**: `(# joints, # timesteps, # dimensions)`
- **Joints**: 55 points on the dancer's body
- **Timesteps**: 5,000-10,000 frames per sequence
- **Dimensions**: 3 (x, y, z coordinates)

### Challenge: Joint Interpretation

**Problem**: Interpreting what each of the 55 joints represents on the human body.

**Solution**: Analyzed reference code to identify semantic meanings of points and created a mapping of 53 relevant joints:

```python
point_labels = [
    'ARIEL', 'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH', 'LBWT', 'LELB', 'LFHD', 'LFRM',
    'LFSH', 'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5',
    'LOHAND', 'LOWR', 'LSHN', 'LTHI', 'LTOE', 'LUPA', 'MBWT', 'MFWT', 'RANK', 'RBHD',
    'RBSH', 'RBWT', 'RELB', 'RFHD', 'RFRM', 'RFSH', 'RFWT', 'RHEL', 'RIEL', 'RIHAND',
    'RIWR', 'RKNE', 'RKNI', 'RMT1', 'RMT5', 'ROHAND', 'ROWR', 'RSHN', 'RTHI', 'RTOE',
    'RUPA', 'STRN', 'T10'
]
```

### Technology: PyVista

**Why PyVista?** Selected for its:
- Superior performance with large datasets
- Advanced 3D visualization capabilities
- Flexible customization options

**Alternatives Considered**: Matplotlib, Mayavi, and Plotly were evaluated but rejected due to performance limitations or rendering inefficiency.

### Skeleton Construction

Created a coherent skeleton by defining connections between joint pairs:

```python
skeleton_lines = [
    (('LHEL', 'LTOE')), (('RHEL', 'RTOE')), (('LMT1', 'LMT5')), (('RMT1', 'RMT5')),
    (('LHEL', 'LMT1')), (('LHEL', 'LMT5')), (('RHEL', 'RMT1')), (('RHEL', 'RMT5')),
    (('RFWT', 'MFWT')), (('LFWT', 'MFWT')), (('LFSH', 'RFSH'))
    # Additional connections added for completeness
]
```

![Skeleton Visualization](https://github.com/user-attachments/assets/9ad3ac08-d6d8-4cd3-9925-921fb1dde673)

### Visualization Improvements

**Challenge 1**: Difficult to determine dancer orientation.

**Solution**: Implemented color differentiation, using red to highlight front-facing parts.

![Color-Coded Skeleton](https://github.com/user-attachments/assets/405baf33-37f3-422f-b822-c775aaf59eef)

**Challenge 2**: Dancer moving out of camera frame during dynamic sequences.

**Solution**: Dynamically adjusted camera position based on skeleton movement, ensuring the dancer remains visible.

### Performance Optimization

**Initial Performance**: ~1 minute to process 1,000 frames.

**Optimizations Applied**:
- Vectorized operations instead of loops
- Improved memory management
- Reduced redundant calculations

**Result**: 83% reduction in rendering time (to ~10 seconds per 1,000 frames).

## Component 2: Motion Autoencoder

### Architecture

The autoencoder learns compact representations of dance sequences:

![Autoencoder Architecture](https://github.com/user-attachments/assets/8996b166-2d2e-4dd7-9bf7-7e00b7ffaf2f)

- **Input**: Dance sequences of shape `[80, 50, 159]` (80 sequences, 50 timesteps, 159 features)
- **Encoder**: Three linear layers with ReLU, reducing to 64 dimensions
- **Decoder**: Three linear layers with ReLU, reconstructing the original sequence

### Data Preprocessing

1. Slice timesteps from 4000 to 8000 to focus on relevant motion segments
2. Select 53 out of 55 joints, excluding indices 26 and 53
3. Invert the Z-axis to align with the original data orientation
4. Reshape into 80 sequences of 50 timesteps each
5. Flatten joint coordinates into a feature vector of size 159 (53 joints × 3 dimensions)

![Data Processing Flow](https://github.com/user-attachments/assets/217ee502-f080-4d52-aaef-73e1fdf02172)

### Training

- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation**: Qualitative assessment of reconstructed sequences

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4ba78c24-c879-4fe1-a9e7-673e3ab9b112" width="250">
      <br>
      <b>Original Sequence</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6e893f86-fb1d-4349-a4b0-fef8d6cb1526" width="250">
      <br>
      <b>Reconstructed Sequence</b>
    </td>
  </tr>
</table>

## Component 3: Semi-Supervised Labeling

### Challenge

Limited labeled data for training a text-to-dance system.

### Approach

1. **Manual Labeling**: Created text descriptions for 10 out of 80 sequences
   - Sample labels: "spin with right hand above", "neck rotation", "kick with left leg"

2. **Data Augmentation**:
   - **Time warping**: Speed variation 0.9–1.1× 
   - **Noise injection**: Standard deviation 0.02
   - Resulted in 5× more training samples

3. **KNN Classification**:
   - Used extracted embeddings from the pre-trained autoencoder
   - Parameters: k=2 with distance weighting
   - Predicted labels for the remaining 70 sequences

### Evaluation

- **t-SNE Visualization**: Colored by predicted labels to assess clustering quality
- **Confidence Scoring**: Used KNN's `predict_proba` to identify uncertain predictions

<table>
  <tr>
    <td><img width="313" src="https://github.com/user-attachments/assets/49528b5e-407a-4b07-92c3-90cf49725645" /></td>
    <td><img width="761" src="https://github.com/user-attachments/assets/c7fb0a28-76df-48fb-b592-2eff5f5b51ac" /></td>
  </tr>
</table>

## Component 4: Text-to-Dance & Dance-to-Text Generation

### Text Mapper Model

A neural network that aligns text and motion embeddings:

- **Input**: 768-dimensional BERT embeddings (`bert-base-uncased`)
- **Output**: 64-dimensional latent vectors compatible with the motion autoencoder
- **Architecture**: Two linear layers with ReLU and dropout
- **Training**: MSE loss between predicted and actual dance embeddings

### Text-to-Dance Generation

1. Encode text description using BERT
2. Map BERT embedding to motion latent space using TextMapper
3. Decode the latent vector with the autoencoder to generate motion

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9efb009f-bdca-42d3-ad21-749d8755d105" width="250">
      <br>
      <b>Generated from Text: "Neck Rotation"</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7b12e989-9031-4e91-83c9-e3bb4264d8f0" width="250">
      <br>
      <b>Original Neck Rotation Sequence</b>
    </td>
  </tr>
</table>

### Dance-to-Text Generation

1. Encode dance sequence into the latent space
2. Use KNN to find the closest text embedding from a pre-defined set
3. Retrieve the corresponding text description

**Design Decision**: KNN was chosen over a transformer decoder due to the small dataset size (80 sequences), which would likely cause overfitting in more complex models.

## Results & Evaluation

### Autoencoder Performance

The autoencoder successfully learned to reconstruct dance sequences with minimal loss:

![Loss Curve](https://github.com/user-attachments/assets/96c7cfcc-218c-483c-9347-77b71bb25e31)

### Labeling Quality

Comparison of manual vs. semi-supervised labeling:

<div>
    <h4>Manual Labels Only</h4>
    <img width="700" src="https://github.com/user-attachments/assets/abc4a119-24cd-4d77-9489-d023be890ca9" />
    <h4>Manual + Semi-Supervised Labels</h4>
    <img width="700" src="https://github.com/user-attachments/assets/b424d901-09b6-462d-af11-ea3fbc1fb839" />    
    <h4>t-SNE Visualization</h4>
    <img width="700" src="https://github.com/user-attachments/assets/08353f9b-c88d-4bc6-9eaa-857941adf84c" />
</div>

The t-SNE visualization highlights some inconsistencies in labeling. While the semi-supervised approach improved classification coverage, imperfect manual labels resulted in occasional retrieval failures. Nevertheless, the system remained functional despite these challenges.

### Generation Capabilities

1. **Text-to-Dance**: Generated sequences captured core motion qualities but sometimes lacked the nuance of original sequences.
2. **Dance-to-Text**: Retrieved plausible descriptions, limited by the small set of available text labels.

![Labeled Animation](https://github.com/user-attachments/assets/6baba1ae-8a60-4541-ad3f-dd4324b783bf)

## Future Directions

While the current implementation successfully demonstrates the concept, several enhancements could improve performance:

1. **Data Expansion**:
   - Larger dataset with more diverse movements
   - Professional labeling by dance experts

2. **Model Improvements**:
   - Transformer-based sequence encoding for better temporal understanding
   - VAE or diffusion model architecture for more natural motion generation
   - Additional motion quality features beyond raw coordinates

3. **Interface Enhancements**:
   - Interactive web-based visualization
   - Real-time generation capabilities
   - Support for multimodal inputs (audio, video, text)

4. **Creative Applications**:
   - Choreographic assistants for dance composition
   - Educational tools for dance analysis
   - Performance augmentation with generative elements

---

*This project was developed as part of the application for HumanAI GSoC 2025.*

# AI-Enabled Choreography: Dance Beyond Music

This repository contains the code and documentation for the **AI-Enabled Choreography** project, submitted as part of the application process for the HumanAI GSoC 2025. The project leverages motion capture data to generate and label dance sequences using a combination of autoencoders, semi-supervised learning, and multimodal text-to-dance and dance-to-text generation. 

## Project Overview

<img width="398" alt="Project Architecture" src="https://github.com/user-attachments/assets/7f536c1f-68d8-43ab-aae8-5dfa940f6f3b" />

The system works across three key components:

1. **Motion Visualization & Autoencoder** - Transforms raw motion capture data into 3D skeleton animations and learns compact dance representations
2. **Semi-Supervised Labeling** - Labels dance sequences with limited manual input using KNN classification
3. **Text-to-Dance Generation** - Creates bidirectional mappings between text descriptions and dance motions

## Data Processing

- **Input**: Motion capture `.npy` files (55 joints × timesteps × 3D coordinates)
- **Preprocessing**: Selecting 53 relevant joints, standardizing sequences, and creating a coherent skeleton visualization

<img width="334" alt="Data Structure" src="https://github.com/user-attachments/assets/217ee502-f080-4d52-aaef-73e1fdf02172" />

## Motion Autoencoder

The autoencoder compresses dance sequences into a 64-dimensional latent space that captures essential movement characteristics.

<img width="800" alt="Autoencoder Architecture" src="https://github.com/user-attachments/assets/8996b166-2d2e-4dd7-9bf7-7e00b7ffaf2f" />

**Reconstruction Quality:**

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

## Semi-Supervised Labeling

With only 10 manually labeled sequences, we label the remaining 70 using KNN classification with data augmentation:

<img width="327" alt="image" src="https://github.com/user-attachments/assets/0cef1b44-fcc2-4fcb-b32d-a6abb9a3be1b" />


t-SNE visualization shows the clustering of similar movements:

<img width="500" alt="t-SNE Visualization" src="https://github.com/user-attachments/assets/08353f9b-c88d-4bc6-9eaa-857941adf84c" />

## Text-to-Dance Generation

Our system can:
1. Generate dance sequences from text descriptions
2. Describe dance sequences with appropriate text labels (using KNN)

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

## Results

Our approach successfully creates:
- High-quality 3D skeleton visualizations with orientation indicators
- Accurate dance sequence reconstructions via autoencoder
- Reasonable text-to-motion and motion-to-text mappings despite limited labeled data

![Dance Animation](https://github.com/user-attachments/assets/6baba1ae-8a60-4541-ad3f-dd4324b783bf)

## Future Work

- Enhanced visualization with shadows and velocity indicators
- Larger dataset with professional labeling
- Transformer-based sequence models for more complex choreography
- Support for multi-dancer choreography using Graph Neural Networks

## Key Insights

> "AI should support artists, not replace them, using intuitive tools like latent spaces for poses and sequences."

- Semi-supervised learning works well with minimal labeled data
- Latent space embeddings effectively capture dance dynamics
- Simple KNN approaches can be effective for small datasets before scaling to transformers

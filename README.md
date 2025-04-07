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
<img width="540" alt="augmentation_concept" src="https://github.com/user-attachments/assets/8c404e79-0e2c-4474-88c7-332d06bf5a16" />


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
       <img src="https://github.com/user-attachments/assets/7919b7e0-0c55-4bd8-a4b5-3aaba878a473" width="250">
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

Unable to do very accurately :
- text-to-motion and motion-to-text mappings because most probably of unprofessional labeling

<img width="520" alt="image" src="https://github.com/user-attachments/assets/a66cf065-7b58-4224-8722-b1f9b64cfdf3" />

![spin with right hand above](https://github.com/user-attachments/assets/f65e03cd-7e5f-47fc-9bd9-1e917e60b079)

for this animation the correct label was spin with right hand above and our model predicted spin with little hand movement
I guess I should not have labelled the images with long text like these actually i thought that bert's embedding would work fine even if the labels were quite different as it would capture the sementic meanings but still nothing like that happened.

## Future Work

- Enhanced visualization with shadows and velocity indicators
- Improving alot the text to dance and dance to text multi model
- Transformer-based sequence models for more complex choreography
- Support for multi-dancer choreography using Graph Neural Networks

## Key Insights

> "AI should support artists, not replace them, using intuitive tools like latent spaces for poses and sequences."

- Semi-supervised learning works well with minimal labeled data
- Latent space embeddings effectively capture dance dynamics
- Simple KNN approaches can be effective for small datasets before scaling to transformers
  
## Note  

- If i am being honest what actually happened was that after making alll the documentation and all when i was making the proposal i cam to know that i actually used the training data itself instead of val while text to dance and dance to text multimodal testing so thats why my initial results were too good but then i realised that the models were not working as expected so i fixed the readme.md with the correct result but still in some documentation if you found that the model worked very great please ignore as it was on trainin dataset.

- Each task has detailed documentation available in its respective folder. Please refer to the following files for more information:  
  - [First Documentation](src/dance_animation/documentation_01.md)  
  - [Second Documentation](src/%20train_dance_text_model/documentation_02.md)
  - [Report in PDF](choreo_ai_report.pdf)  

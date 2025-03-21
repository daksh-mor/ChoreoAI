# AI-Enabled Choreography: Dance Motion Generation and Labeling

The implementation is divided across three Jupyter notebooks, each focusing on a distinct component of the workflow.

> "I didn’t want to imitate anybody. Any movement I knew, I didn’t want to use."  
> — Jennings (2009)

## Table of Contents
1. [Insights from Research Papers](#insights-from-research-papers)
2. [Approach Overview](#approach-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Motion Autoencoder](#motion-autoencoder)
5. [Semi-Supervised Labeling](#semi-supervised-labeling)
6. [Text-to-Dance and Dance-to-Text Generation](#text-to-dance-and-dance-to-text-generation)
7. [Visualization](#visualization)
8. [Results and Evaluation](#results-and-evaluation)
9. [Conclusion](#conclusion)

---

## Insights from Research Papers

this section combines key takeaways from two papers.

#### Papers Reviewed: [1907.05297](https://arxiv.org/abs/1907.05297) & [2207.12126](https://arxiv.org/abs/2207.12126)
- **Learnings**:
  - AI should support artists, not replace them, using intuitive tools like latent spaces (poses/sequences).
  - Laban Effort labels add variety and artistic control.
  - Semi-supervised learning and data augmentation work well with minimal labeled data.
- **Improvements**:
  - Swap RNNs/LSTMs with Transformers for better sequence modeling.
  - Upgrade to VAEs/GANs or Diffusion Models for sharper, diverse movements.
  - Add continuous, varied artistic labels and support group choreography with Graph Neural Networks (GNNs).

---

## Approach Overview

The project integrates dance motion analysis and natural language processing to create a multimodal system capable of generating dance sequences from text and describing dance sequences with text. It is implemented in three Jupyter notebooks:

1. **Notebook 1: Motion Autoencoder Training and Visualization**
   - Preprocesses motion capture data.
   - Trains an autoencoder to learn 64-dimensional embeddings of dance sequences.
   - Visualizes original and reconstructed motion sequences for qualitative evaluation.

2. **Notebook 2: Semi-Supervised Labeling of Dance Sequences**
   - Extracts embeddings using the pre-trained autoencoder.
   - Manually labels 10 out of 80 sequences, then uses KNN with data augmentation to label the rest.
   - Visualizes embeddings with t-SNE to assess clustering.

3. **Notebook 3: Text-to-Dance and Dance-to-Text Generation**
   - Trains a `TextMapper` model to align BERT text embeddings with the autoencoder’s latent space.
   - Generates dance sequences from text descriptions.
   - Uses KNN to generate text descriptions from dance sequences.

<img width="398" alt="image" src="https://github.com/user-attachments/assets/7f536c1f-68d8-43ab-aae8-5dfa940f6f3b" />


This approach combines unsupervised learning (autoencoder), semi-supervised learning (KNN labeling), and contrastive learning (text-motion alignment) to address the challenge of limited labeled data, as outlined in the raw idea.

---

## Data Preprocessing

- **Input Data**: Motion capture files in `.npy` format with an initial shape of `[# joints, # timesteps, # dimensions]`.
- **Preprocessing Steps**:
  - Load the data and slice timesteps from 4000 to 8000 to focus on relevant motion segments.
  - Select 53 out of 55 joints, excluding indices 26 and 53, using a boolean mask.
  - Invert the Z-axis to align with the original data orientation.
  - Reshape into 80 sequences of 50 timesteps each, flattening joint coordinates into a feature vector of size 159 (53 joints × 3 dimensions), resulting in a tensor shape of `[80, 50, 159]`.
  - Convert the NumPy array to a PyTorch tensor with `torch.float` datatype.
Most of this is done while making the animation

<img width="334" alt="image" src="https://github.com/user-attachments/assets/217ee502-f080-4d52-aaef-73e1fdf02172" />

---

## Motion Autoencoder

- **Purpose**: Learn a compact 64-dimensional latent representation of dance sequences.
- **Architecture**:
  - **Encoder**: Three linear layers with ReLU activations, reducing the input (`50 × 159`) to a 64-dimensional latent space.
  - **Decoder**: Three linear layers with ReLU activations, reconstructing the input from the latent space.
- **Training**: 
  - Uses Mean Squared Error (MSE) loss to minimize reconstruction error.
  - Pre-trained model loaded from `'models/best_autoencoder.pth'` for evaluation.

<img width="1438" alt="image" src="https://github.com/user-attachments/assets/8996b166-2d2e-4dd7-9bf7-7e00b7ffaf2f" />

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

---

## Semi-Supervised Labeling

- **Strategy**:
  - Manually label 10 out of 80 sequences with descriptive terms (e.g., “spin with right hand above”).
  - Use the pre-trained autoencoder to extract 64-dimensional embeddings for all sequences.
  - Augment the labeled data with time warping (speed variation 0.9–1.1) and noise (std 0.02) to increase robustness.
  - Train a KNN classifier (k=2, distance-weighted) on augmented embeddings to predict labels for the remaining 70 sequences.
- **Evaluation**: 
  - Visualize embeddings with t-SNE, coloring points by predicted labels.
  - Compute confidence scores for predictions using KNN’s `predict_proba`.

<table>
  <tr>
    <td><img width="313" src="https://github.com/user-attachments/assets/49528b5e-407a-4b07-92c3-90cf49725645" /></td>
    <td><img width="761" src="https://github.com/user-attachments/assets/c7fb0a28-76df-48fb-b592-2eff5f5b51ac" /></td>
  </tr>
</table>


---

## Text-to-Dance and Dance-to-Text Generation

- **TextMapper Model**:
  - Maps 768-dimensional BERT embeddings (`bert-base-uncased`) to the 64-dimensional latent space of the motion autoencoder.
  - Architecture: Two linear layers with ReLU and dropout, trained with MSE loss to align text and dance embeddings.
- **Text-to-Dance**:
  - Encode a text description (e.g., “kick with left leg”) using BERT and `TextMapper`.
  - Decode the resulting latent vector with the autoencoder to generate a motion sequence.
 
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9efb009f-bdca-42d3-ad21-749d8755d105" width="250">
      <br>
      <b>Made Neck Rotation from Text</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7b12e989-9031-4e91-83c9-e3bb4264d8f0" width="250">
      <br>
      <b>Original Neck Rotation</b>
    </td>
  </tr>
</table>
    
- **Dance-to-Text**:
  - Encode a dance sequence into the latent space.
  - Use KNN to find the closest text embedding from a pre-defined set and retrieve its description.
- **Reasoning**: KNN is chosen over a transformer decoder due to the small dataset (80 sequences), avoiding overfitting and leveraging proximity in the latent space.

<div style="position: relative; display: inline-block;">
    <img src="https://github.com/user-attachments/assets/bee184b8-a914-461c-8604-518ceab27dbd" width="250">
    <p style="position: absolute; top: 10px; left: 10px; background: rgba(0, 0, 0, 0.7); color: white; padding: 5px; font-size: 14px;">
        This GIF's dance text was predicted by the model as "Neck Rotation"
    </p>
</div>



---


## Visualization

- **3D Skeleton Animation**:
  - Custom functions create a PyVista mesh from joint coordinates and connections.
  - Animations are saved as GIFs and MP4s, showing skeletal motion over time.


![label1](https://github.com/user-attachments/assets/6baba1ae-8a60-4541-ad3f-dd4324b783bf)

---

## Results and Evaluation

- **Labeling Performance**:
  - Qualitative evaluation via t-SNE shows reasonable clustering of similar motions.
  - Average KNN confidence scores typically exceed 0.7, with 1 low-confidence predictions indicating labeling uncertainty.
- **Generation Quality**:
  - Dance-to-text retrieves plausible descriptions but is limited by the small set of text labels.
  
<div align="center">
    <p><b>Loss of Autoencoder</b></p>
    <img width="615" src="https://github.com/user-attachments/assets/96c7cfcc-218c-483c-9347-77b71bb25e31" />
</div>

<div align="center">
    <h3>Manual Labelled Only</h3>
    <img width="761" src="https://github.com/user-attachments/assets/abc4a119-24cd-4d77-9489-d023be890ca9" />
    <h3>Manual + Semi-Supervised Learning Labels</h3>
    <img width="750" src="https://github.com/user-attachments/assets/b424d901-09b6-462d-af11-ea3fbc1fb839" />    
    <h3>t-SNE Visualization of the Second Image</h3>
    <img width="864" src="https://github.com/user-attachments/assets/08353f9b-c88d-4bc6-9eaa-857941adf84c" />
    <h3>hence</h3>
    <p style="text-align: justify; max-width: 800px;">
        The t-SNE visualization highlights inconsistencies in labeling. While the semi-supervised learning approach improved classification, 
        poor manual labels resulted in retrieval failures. However, the system was still functional despite these issues, 
        proving that better labeling could further enhance performance.
    </p>
</div>

---

## Conclusion

This project successfully implements an AI-enabled choreography system, integrating motion autoencoders, semi-supervised labeling, and multimodal generation. While the results demonstrate potential—accurate reconstructions, reasonable labels, and text-motion mappings—limitations include the small dataset and reliance on limited manual labels. Future enhancements could involve larger datasets, professional labeling, and advanced generative models like transformers with sufficient data.

---

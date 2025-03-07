# Dance Data Animation Implementation

### Overview
This documentation details my approach to visualizing motion capture dance data as part of the "AI-Enabled Choreography: Dance Beyond Music" project.

<img width="375" alt="image" src="https://github.com/user-attachments/assets/b4209d4d-7bc3-434e-be62-321bf101e32f" />

## Data Understanding
The provided dataset consists of six `.npy` files, each representing different dance sequences from the same performer. 

### Data Structure
- **Shape**: `(# joints, # timesteps, # dimensions)`
- **Joints**: 55 points on the dancer's body
- **Timesteps**: 5,000-10,000 frames per sequence
- **Dimensions**: 3 (x, y, z coordinates)

### Challenge: Joint Interpretation
#### Problem
The dataset includes 55 joints, which initially presented a challenge in understanding what each point represents on the human body and how they should be connected to create a coherent skeleton visualization.
<img width="695" alt="image" src="https://github.com/user-attachments/assets/02890a14-601c-434b-977f-5028ad765d30" />

#### Solution
I analyzed reference code examples and identified the semantic meaning of each joint. I created a mapping of 53 relevant points (removing 2 redundant points at indices `[26, 53]` that were nearly overlapping with others):

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

This mapping allowed me to understand what each joint represents (e.g., `LKNE` = left knee, `RFHD` = right front head).

## Technology Selection: PyVista

### Why PyVista?
PyVista was selected for its superior performance with large datasets, advanced 3D visualization capabilities, and flexible customization options.

### Alternative Libraries Considered
(Matplotlib, Mayavi, Plotly) were evaluated but rejected due to performance limitations, complexity, or rendering inefficiency.

## Implementation Details

### 1. Skeleton Construction
I defined the skeleton by specifying connections between joint pairs, creating lines that represent limbs and body segments:

```python
skeleton_lines = [
    (('LHEL', 'LTOE')), (('RHEL', 'RTOE')), (('LMT1', 'LMT5')), (('RMT1', 'RMT5')),
    (('LHEL', 'LMT1')), (('LHEL', 'LMT5')), (('RHEL', 'RMT1')), (('RHEL', 'RMT5')),
    (('RFWT', 'MFWT')), (('LFWT', 'MFWT')), (('LFSH', 'RFSH'))
]
```

**Enhancement**: I added additional joint connections beyond the reference implementation to create a more complete and symmetric skeleton representation.
<img width="857" alt="image" src="https://github.com/user-attachments/assets/9ad3ac08-d6d8-4cd3-9925-921fb1dde673" />

### 2. Visualization Improvements
#### Problem
In initial visualizations, it was difficult to determine which direction the dancer was facing.

#### Solution
I implemented color differentiation, using red to highlight the front-facing parts of the skeleton. This enhanced the viewer's ability to interpret the dancer's orientation and movement direction.

<img width="1265" alt="image" src="https://github.com/user-attachments/assets/405baf33-37f3-422f-b822-c775aaf59eef" />

#### Problem
The dancer would occasionally move out of the camera frame during dynamic sequences.

#### Solution
I analyzed the frames to check if the dancer was going out of view and adjusted the camera accordingly, keeping it static and ensuring the dancer remains in the frame.


### 3. Performance Optimization
#### Problem
Initial rendering was slow, taking approximately 1 minute to process 1,000 frames.

#### Solution
I applied several optimization techniques:
- **Vectorized operations** instead of loop-based processing
- **Optimized memory management** for large datasets
- **Reduced redundant calculations** in the rendering pipeline

These improvements reduced rendering time by ~83%, bringing the processing time down to approximately 10 seconds per 1,000 frames.

## Workflow Summary
1. **Data loading**: Read and pre-process the `.npy` motion capture files.
2. **Skeleton definition**: Map joint points to create a coherent human figure.
3. **Animation setup**: Configure PyVista for efficient rendering.
4. **Visualization enhancements**: Add orientation indicators.
5. **Output generation**: Export the animation as either GIF or MP4.

## Results
The implementation successfully visualizes the dance sequences as smooth 3D animations, with clear skeletal structure and orientation indicators. The optimized rendering pipeline allows for quick generation of visualizations, enabling efficient analysis of different dance sequences.

Output formats include both GIF and MP4, allowing for flexible usage in different contexts (documentation, presentations, further analysis).

## Future Improvements
While the current implementation meets project requirements effectively, potential enhancements could include:
1. Adding shadow effects to improve depth perception.
2. Implementing color gradients to represent velocity or acceleration.
3. Creating a user interface for interactive parameter adjustment.
4. Adding visualization options for joint trajectories over time.

These improvements would be valuable for detailed motion analysis but were not implemented in this phase to maintain focus on the core visualization requirements.

## Conclusion
The visualization component provides a solid foundation for the "AI-Enabled Choreography" project by enabling clear, efficient representation of complex motion capture data. The optimized implementation balances performance with visual clarity, creating a tool that will support the subsequent machine learning phases of the project.


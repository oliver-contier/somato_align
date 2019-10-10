#  somato-align

Cross-subject functional alignment of somatosensory digit representations in 7T fMRI data. This project is part of Oliver's lab rotation at INM-7. 

## Research question
 
Representations of body parts in primary somatosensory cortex follow some roughly characterized anatomical tophography. However, aligning these ideosyncratic somatosensory maps anatomically (e.g. onto a standard template) may lead to misplacement and distortion. Small body parts especially might not be tangible with standard-resolution data (e.g. acquired with 3T fMRI) and especially prone to such distortions. Additionally, somatosensory representations might be represented in a distributed fashion and thus be inacessible by mere anatomical alignment. 

This project aims to identify potentially distributed response patterns single digits and align these individual representations across  participants. 
 
## Dataset description

Details will follow.

## Analysis rationale

 - Subject-level ICA: Extract individual response components and systematically identify the subset that might correspond in their temporal characteristics to the digit stimulation procedure.
 - Shared response model (alternatively Hyperalignment): Project these stimulus-specific response vectors from their individual voxel space into a shared space.
 - Validation: Cross-subject classification?
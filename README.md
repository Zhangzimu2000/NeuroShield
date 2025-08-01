# NeuroShield
This repository provides a comprehensive framework for analyzing, detecting, and mitigating adversarial attacks on neural networks. It includes utilities for identifying suspicious neurons, fine-tuning models to improve robustness, and evaluating the modified models. The project is built with PyTorch and integrates popular adversarial-attack libraries such as torchattacks.

## Features
* Adversarial Example Generation
  + Fast Gradient Sign Method (FGSM)
  + Projected Gradient Descent (PGD)
* Suspicious Neuron Detection
  + Identifies neurons with abnormal activations in response to adversarial inputs
  + Tools for layer- and channel-level analysis
* Fine-Tuning and Repair
  + Fine-tunes specific layers to mitigate adversarial effects
  + Masks or modifies weights of channels with the highest anomaly scores
* Evaluation
  + Measures model performance on both clean and adversarial inputs
  + Assesses the impact of weight masking and fine-tuning on overall accuracy

## Repository Structure
* train.py Implements training and fine-tuning routines to repair suspicious neurons and improve adversarial robustness. Key functions:
  + fine_tune_model: Fine-tunes layers with high anomaly counts
  + evaluate_model: Evaluates accuracy on clean and adversarial datasets
* test.py Tests a pre-trained model’s performance against adversarial attacks. Key components:
* 

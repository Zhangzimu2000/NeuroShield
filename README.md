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
  + Loads pre-trained architectures (ResNet18, ResNet34, ResNet50, etc.)
  + Uses evaluate_model from train.py for adversarial and clean accuracy
* utils.py Utility functions for model analysis and layer-wise operations. Key functions:
  + get_model_layers: Extracts all valid layers from a model
  + get_layer_output: Captures intermediate outputs of specified layers
  + get_anomaly_neurons: Identifies neurons with abnormal activations via thresholding or statistical methods
* neuron.py Focuses on neuron-level anomaly detection and handling. Key components:
  + Custom ImageDataset for loading labeled datasets
  + process_batch: Identifies misclassified or adversarial samples
  + save_results: Exports evaluation results to JSON

## Example Workflow
1. Fine-tune a model and detect anomalies
  * Run train.py to fine-tune a pre-trained ResNet50 on CIFAR-10.
  * Monitor anomaly statistics printed during training.
2. Test model robustness
  * Execute test.py to evaluate performance on clean and adversarial inputs.

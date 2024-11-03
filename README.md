# CSRNet Implementation for Crowd Counting

This repository contains an implementation of CSRNet (Crowd Counting via Regression Networks) for estimating crowd density using the Shanghai dataset. The model has been trained and evaluated on both Part A and Part B of the dataset.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)


## Introduction

CSRNet is a state-of-the-art model for crowd counting, which utilizes a regression-based approach to predict density maps from images. The implementation in this repository is based on the architecture described in the research paper linked below.

## Requirements

To run this project, you will need the following Python packages:

- torch
- torchvision
- h5py
- numpy
- PIL
- matplotlib
- json

You can install the required packages using the provided `requirements.txt`.

## Dataset

The model is trained using the Shanghai dataset, which consists of two parts:
- **Part A:** Contains 482 images.
- **Part B:** Contains 316 images.

The ground truth density maps are provided in `.h5` format.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/csrnet-crowd-counting.git
   cd csrnet-crowd-counting

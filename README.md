# mobilepipe-isef

## Overview

This is the presentation code repository for the ISEF 25-26 Project: "MobilePipe: iPhones as Parallel Compute Accelerators for Local Machine Learning", by Alexander Chen (Homeschool, Dallas, TX, USA).

## Repository Structure

This repository contains code snippets for experiment infrastructure implementation (host, client, and communication layers, experiment run workflow) and all result logs from the 96 experiment test runs. 

### Experimentation Infrastructure

* `client` - Client Layer: mobile (iOS) layer using Swift + Metal Performance Shaders (MPSGraph for hardware accelerated ML operations)
* `comms` - Communication Layer: libimobiledevice iproxy & data communication handler
* `host` - Host Layer: desktop Python layer using PyTorch + torchvision
* `workflow` - Experiment Workflow: training scripts and CLI tools for running automated train workflows

### Experiment Results

* `results` - Results: logs from dynamic pipeline configuration selection and complete runs; organized by experiment configuration (host type, centralized/decentralized, batchsize configuration, microbatch size configuration), and consistency run index (1, 2, or 3).

## Development Workflow
* Log result parsing/formatting & visualization python scripts were mainly written through qwen-code, with some detail corrections by hand (using model Qwen3-Coder-480B-A35B-Instruct).
* Repetitive parts of the manual ResNet layer MPSGraph implementation was partially generated via Vim macro scripts.
* The rest of the codebase was written by hand (even top closed-source models (tested w/ GPT-5, Gemini 3) struggled significantly when attempting to handle the Python + Swift environment alongside the more obscure MPSGraph framework, most likely due to the uniquely unfamiliar working environment and context-switching.

## Usage

This repository is in format for presentation, not ready to be executed as is.
However all needed source code to run experiments is contained in this repository.
`workflow`, `host`, and Python `comm` scripts should be loaded and run from the host computer.
`client` and Swift `comms` files should be loaded in an iOS application (see Apple Xcode iOS development).
When running experiments, ensure the iOS client device is wired to the host computer device and connection permissions have been allowed.
Also ensure that the `iproxy` port forwarding script is running on the desktop before running `mobilepipe` training experiments.


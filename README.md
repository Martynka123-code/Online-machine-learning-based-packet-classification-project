# Online-machine-learning-based-packet-classification-project

![Build Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Topic](https://img.shields.io/badge/area-Network%20Intelligence-orange)

## Project Overview

OMLBPC Project is a high-performance framework designed for **real-time network traffic classification**. The core objective is to identify the specific application or service type (e.g., YouTube, Microsoft Teams, Netflix) by analyzing live packet streams directly from the network interface.

Our project evaluates the trade-offs between traditional statistical features and modern deep learning techniques in an **online inference** environment.

## Key Features

* **Live Packet Capture:** Real-time data acquisition using `Scapy` / `PyShark`.
* **Dual-Path Classification:**
  * **Flow-based:** Analyzes aggregate features like packet timing, lengths, and burstiness using traditional ML models (e.g., Random Forest).
  * **Byte-level:** Raw payload analysis using 1D/2D Convolutional Neural Networks (CNN) to capture spatial patterns in packet headers and data.
* **Comparative Analysis:** Performance benchmarking between statistical and deep learning methods in a live environment.

## Classification Methodology

To achieve robust results, the system implements two parallel processing pipelines:

### 1. Flow-based Classification (Aggregate Analysis)

* **Approach:** Groups packets into flows (5-tuple: Src/Dst IP, Src/Dst Port, Protocol).
* **Feature Set:** Statistical aggregates such as Flow Duration, Inter-Arrival Time (IAT), Mean Packet Length, and TCP window flags.
* **Model:** Decision-trees/Gradient Boosting (e.g., Random Forest).

### 2. Byte-level Classification (Payload Analysis)

* **Approach:** Processes raw packet data "byte-by-byte" without manual feature extraction.
* **Feature Set:** Raw header and payload bytes (first $N$ bytes of a packet/session).
* **Model:** **Convolutional Neural Networks (CNN)** designed to capture spatial dependencies in the packet structure, similar to computer vision tasks.

## Classification Targets

The models are trained to identify:

* **Streaming:** YouTube, Netflix, Twitch
* **Communication:** Microsoft Teams, Zoom, Slack
* **Web/Standard:** HTTPS, Mail (IMAP/SMTP), DNS
* **File Transfer:** FTP, P2P (if applicable)

## Tech Stack

* **Language:** Python
* **Capture:** Scapy / Libpcap
* **ML/DL:** TensorFlow, PyTorch, Scikit-learn
* **Data Handling:** Pandas, NumPy

## Literature & References

Every article and source, which inspired us and gave us required knowledge to work with this project is in LITERATURE.md file.

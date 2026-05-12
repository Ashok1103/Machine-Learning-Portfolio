# Machine Learning Portfolio
Ashok Kumar Ramu | Physics and Astronomy, University of Waterloo
 
---
 
## About
 
I'm a Physics and Astronomy student at the University of Waterloo with hands-on industry experience in data analysis and machine learning through co-op. My background in computational physics gives me a strong foundation in numerical methods, statistical modelling, and writing clean, reproducible code, which I apply directly to ML problems.
 
This portfolio collects my applied machine learning projects. Each project includes notebooks, results, and documentation. Datasets are either linked externally or omitted where confidential.
 
---
 
## Projects
 
### 1. WatSPEED Enrollment Prediction
`Classification · Random Forest · Logistic Regression · Feature Engineering · K-Means Clustering`
 
Built an end-to-end ML pipeline to predict whether a professional learner will remain enrolled or drop out of a continuing education program. Completed as a final deliverable during my Market Research Analyst co-op at WatSPEED, University of Waterloo.
 
- Merged and cleaned enrollment data across 3 programs and multiple years
- Engineered features from raw job titles (491 unique values) using rule-based parsing
- Trained and compared Logistic Regression (baseline) and Random Forest (main model) with class imbalance handling
- Applied K-Means clustering to identify distinct learner segments
- Built modular `src/` pipeline separating preprocessing, training, and evaluation
**[View Project](https://github.com/Ashok1103/Machine-Learning-Portfolio/blob/main/WatSPEED%20Enrolment%20Prediction)**
 
---
 
### 2. Galaxy Morphology Classifier
`Deep Learning · Computer Vision · PyTorch · ResNet-18 · Grad-CAM · Gradio`
 
Trained a galaxy morphology classifier on 17,736 images from the DECam Legacy Survey, progressing from a NumPy MLP built from scratch to a fine-tuned ResNet-18 reaching 90% test accuracy across three morphological classes: Smooth/Elliptical, Spiral, and Edge-on/Disturbed.
 
- Implemented backpropagation and mini-batch SGD from scratch in NumPy (no PyTorch)
- Built a custom 4-block CNN in PyTorch, then fine-tuned ResNet-18 using two-stage transfer learning
- Used Grad-CAM to visualise model attention, confirming the model detects disk geometry and dual nuclei in merging systems rather than just central brightness
- Misclassified edge-on galaxies reveal a real astrophysical degeneracy: projection effects make edge-on disks visually similar to smooth ellipticals at this resolution
- Deployed as a live Gradio app on Hugging Face Spaces
| Model | Parameters | Test Accuracy |
|---|---|---|
| NumPy MLP | 1,082,115 | 75.8% |
| Custom CNN | 422,179 | 87.8% |
| ResNet-18 fine-tuned | 11,178,051 | 90.0% |
 
**[View Project](https://github.com/Ashok1103/Machine-Learning-Portfolio/blob/main/Galaxy%20Classifier)** | **[Live Demo](https://ashok1103-galaxy-morphology-classifier.hf.space)**
 
---
 
## Skills
 
- **Languages:** Python
- **Deep Learning:** PyTorch, torchvision, ResNet, transfer learning, Grad-CAM, CNNs, backpropagation
- **ML:** scikit-learn, Random Forest, Logistic Regression, K-Means, feature engineering, class imbalance handling
- **Data:** pandas, numpy, h5py, openpyxl, joblib
- **Visualization:** matplotlib, seaborn
- **Deployment:** Gradio, Hugging Face Spaces
- **Workflow:** Jupyter Lab, modular src/ architecture, Git
---
 
## Contact
 
Feel free to reach out via GitHub for questions about any of these projects.

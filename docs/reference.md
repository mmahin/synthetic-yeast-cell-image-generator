# References

## Synthetic Image Generation

1. **OpenCV Documentation**  
   OpenCV provides tools for image processing and manipulation, used here for drawing synthetic yeast cells and masks.  
   [OpenCV Documentation](https://docs.opencv.org/)

2. **NumPy Documentation**  
   NumPy is used extensively for numerical operations, such as generating random numbers for cell positioning and size.  
   [NumPy Documentation](https://numpy.org/doc/)

3. **Pillow (PIL) Library**  
   Pillow is used for image manipulation and saving generated images in different formats.  
   [Pillow Documentation](https://pillow.readthedocs.io/)

## Mask R-CNN Model

1. **Mask R-CNN Research Paper**  
   Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross B. Girshick, “Mask R-CNN,” 2017. This paper details the Mask R-CNN architecture for object detection and instance segmentation.  
   [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)

2. **PyTorch Documentation**  
   PyTorch provides modules for implementing deep learning models. Here, it is used to implement Mask R-CNN and handle data processing for training and evaluation.  
   [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

3. **Torchvision Library**  
   Torchvision contains pre-trained models and utilities that are compatible with PyTorch, including a built-in implementation of Mask R-CNN.  
   [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)

## Graphical User Interface

1. **Tkinter Documentation**  
   Tkinter is the standard GUI library for Python, used here to create an interface for setting generation parameters, initiating model training, and visualizing results.  
   [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)

## Data and Training Management

1. **DataLoader and Dataset Documentation (PyTorch)**  
   DataLoader and Dataset classes from PyTorch handle data processing and batching for training and testing.  
   [Data Loading and Processing Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

2. **Adam Optimizer**  
   Kingma, D. P., & Ba, J. (2014). “Adam: A Method for Stochastic Optimization.” This is used as the optimizer in training Mask R-CNN in the project.  
   [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)

## Additional References

1. **Intersection over Union (IoU) Metric**  
   IoU is a standard evaluation metric for object detection and segmentation models. It measures the overlap between the predicted and ground truth masks, which helps evaluate Mask R-CNN’s performance.  
   [Intersection over Union (IoU) Explained](https://en.wikipedia.org/wiki/Jaccard_index)

2. **Instance Segmentation with Mask R-CNN - PyTorch Tutorial**  
   A practical tutorial on implementing instance segmentation using Mask R-CNN with PyTorch and Torchvision.  
   [Instance Segmentation with PyTorch](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

---

## Major Contributor

1. **ChatGPT by OpenAI**  
   ChatGPT by OpenAI assisted extensively in structuring the project, debugging, documentation, and providing implementation support throughout the development of this application.  
   [OpenAI - ChatGPT](https://openai.com/chatgpt)

---

## Suggested Further Reading

- **“Deep Learning” by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**  
  This book provides a comprehensive foundation in deep learning concepts, useful for understanding the principles behind Mask R-CNN and other deep learning architectures.

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron**  
  This book provides practical insights into building machine learning and deep learning models, including tips on training and evaluating models.

---

This `references.md` document consolidates foundational references for key technologies, algorithms, and methodologies used in the project.

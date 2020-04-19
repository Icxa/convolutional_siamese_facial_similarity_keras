# How similar are these two faces?
Facial geometry and structure is _fascinating_. Studies come out every now and then linking certain facial characteristics or structure archetypes to certain behavioral trends (google 'men wider faces aggressive' or check out https://royalsocietypublishing.org/doi/10.1098/rsos.181552) for an idea of what our faces can help reveal about us.
I want to employ deep learning to analyze faces in all sorts of ways. But that's quite complex, so to begin, I'm constructing a comparative 'facial similarity' tool using Siamese convolutional neural networks. 

How do we get a net to judge similarity? The most direct way is to feed it two faces, and teach it to distinguish congruent or different identities.
- Tools used:
    - Keras
    - NumPy
    
## Siamese CNN: Are these two faces different people, or the same person at different angles?
Think of a Siamese net as two identical-twin networks lined up side by side; same architecture, same weights. They take one image each, run it through convolutional layers, and output a flattened feature vector.
By comparing these two vectors in a contrastive loss function, we can garner something similar to similarity.
(In practice, we just take one model and feed it two images in succession, but the original idea has a catchy name, so it stuck.)


The current net iteration determines whether the pair is different photos of the same person, or different people entirely. Using 2000 RGB images from UTKFaces (https://susanqq.github.io/UTKFace/), we feed the net same or different pairs and backprop accordingly.
By training it in this way, once it reaches a satisfactory accuracy, we can remove the top 'same or dif' output layer to access the flattened, combined contrastive loss output underneath, which will be a decimal roughly between 0 and 4 measuring 'dissimilarity' (which we can invert).

## The current iteration of the net reaches ~67% test accuracy in 10 epochs with early stopping.
- Limitations:
    - data variety
      - UTKFaces only has one image per person ('class'), so I used image augmentation to randomly alter each photo 8 times over
        - this allows the net to be trained, but raises the question: is image warping a valid substitute for actual photos at other angles?
    - architecture complexity
      - a denser net with larger feature vector layers (4096 instead of 2048, etc) would likely perform better
      - but running on a local small-scale GPU presents runtime issues
      
- Next steps:
  - Run with AT&T faces: 40 identities, grayscale, ~7 images per identity
    - compared to a traditional classifier CNN, for a similarity network such as this, images_per_class is more important than number_of_classes
  - Load model into SageMaker to reduce run load
  - Experiment with transfer learning (VGGnet, FaceNet)
  
### Blog posts
Overview
https://medium.com/@mark.s.cleverley/face-identification-siamese-convolutional-neural-nets-b4c66771595c

### Structural concepts borrowed from:
Harshvardhan Gupta
https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
Florencia Leoni
https://github.com/fpleoni/its_all_in_the_family




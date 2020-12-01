# Certified Interpretability Robustness for Class Activation Mapping

This repository contains code for the NeurIPS 2020 ML4AD Workshop paper , "Certified Interpretability Robustness for Class Activation Mapping."

Interpreting machine learning models is challenging but crucial for ensuring the safety of deep networks in autonomous driving systems. Due to the prevalence of deep learning based perception models in autonomous vehicles, accurately interpreting their predictions is crucial. While a variety of such methods have been proposed, most are shown to lack robustness. Yet, little has been done to provide certificates for interpretability robustness. Taking a step in this direction, we present CORGI, short for Certifiably prOvable Robustness Guarantees for Interpretability mapping. CORGI is an algorithm that takes in an input image and gives a certifiable lower bound for the robustness of the top $k$ pixels of its CAM interpretability map. We show the effectiveness of CORGI via a case study on traffic sign data, certifying lower bounds on the minimum adversarial perturbation not far from (4-5x) state-of-the-art attack methods.

# Results
This image shows a sample image being attacked with 3 different radii. Note that under the CORGI certifiable guarantee, the interpretation is the same.

<a href="https://ibb.co/vPGNXgz"><img src="https://i.ibb.co/bBZ4bqd/attackbig.png" alt="attackbig" border="0"></a>

This image shows a comparison between CORGI lower bounds and attack bounds. For our network, CORGI bounds are only 4-5x away from optimal attack bounds.

<a href="https://ibb.co/rFVybSB"><img src="https://i.ibb.co/7jqyg53/corgi.png" alt="corgi" border="0"></a>


# &#948;NN: Discontinuous Neural Networks

by [Francesco Della Santa](https://www.researchgate.net/profile/Francesco-Della-Santa) and [Sandra Pieraccini](https://www.researchgate.net/profile/Sandra-Pieraccini).

In this repository, we publish the code used to implement the Discontinuous Neural Networks (&#948;NNs) presented in the paper
_Discontinuous Neural Networks and Discontinuity Learning_, available soon in Journal of Computational and Applied Mathematics (accepted for publication).

In the article we define a novel typology of Neural Network layers endowed with
new learnable parameters and discontinuities in the space of the activations.
These layers allow to create a new kind of Neural Networks, whose main property
is to be discontinuous, able not only to approximate discontinuous functions
but also to learn and detect the discontinuity interfaces.

The main idea behind learnable discontinuities for NNs is to apply the effects
of a bias "outside" the activation function only when the inputs are non-negative (i.e., using the Heaviside 
_H_ function). 
Thanks to this new bias, the NN has a new trainable parameter that introduces a 
discontinuity in the function of the layer and, then, in the function of the NN. 
The discontinuity introduced depends both on the new
parameters and on the weights and biases of the NN; for this reason we will
refer to learnable discontinuities.

![Examples of discontinuous activation functions (i.e., 
"classic" a.f. plus a multiple of Heaviside](https://i.imgur.com/dbHNZAu.jpg)
Fig. 1: Examples of discontinuous activation functions (i.e., 
"classic" a.f. plus a multiple of _H_.

## Table of Contents
- [License](https://github.com/Fra0013To/deltaNN/blob/main/README.md#license)
- [Requirements](https://github.com/Fra0013To/deltaNN/blob/main/README.md#requirements)
- [Getting Started](https://github.com/Fra0013To/deltaNN/blob/README.md#getting-started)
  - [Inputs/Outputs Description](https://github.com/Fra0013To/deltaNN/blob/main/README.md#inputsoutputs-description)
  - [Layer Initialization](https://github.com/Fra0013To/deltaNN/blob/main/README.md#layer-initialization)
  - [Run the Example](https://github.com/Fra0013To/deltaNN/blob/main/README.md#run-the-example)
- [Citation](https://github.com/Fra0013To/deltaNN/blob/main/README.md#citation)

## License
_deltaNN_ is released under the MIT License (refer to 
the [LICENSE file](https://github.com/Fra0013To/deltaNN/blob/main/LICENSE) for details).

## Requirements
- matplotlib 3.4.2
- numpy 1.19.5
- pandas 1.2.4
- PyYAML 5.4.1
- scikit-learn 0.24.2
- scipy 1.6.3
- tensorflow 2.4.1

**N.B.:** in the requirements we use tensorflow for CPUs but the codes work also with tensorflow for GPUs.

## Getting Started
The discontinuous layer can be used and added to a Keras model as any other Keras layer. 
In the following, we describe the inputs and outputs of a discontinuous layer and we list the arguments for a 
discontinuous layer initialization. 
Similar information is contained in the class code as comments/helps.

Then, we describe the python scripts uploaded on this repository to replicate the "acetone example" 
of the paper (_Real Data Test Case_ section). In these scripts is possible to see how to build, 
train, and analyze a discontinuous NN.

### Layer Initialization
The _DiscontinuityDense_ class, in [nnlayers module](https://github.com/Fra0013To/deltaNN/blob/main/nnlayers.py) 
of this repository, is defined as a subclass of [_tensorflow.keras.layers.Dense_](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense). 
Then, we list and describe only the new input arguments for the initialization. All the other arguments 
(e.g., _activation_, _kernel_initializer_, etc.) are inherited by the _Dense_ class.

- **discontinuity_initializer**: keras initializer for trainable discontinuity jump parameters
(default zeros).


### Run the Examples
To see a code example of &#948;NN construction, training and analysis, see the scripts 
[example_train_acetone.py](https://github.com/Fra0013To/deltaNN/blob/main/example_train_acetone.py) 
and 
[example_analyze_acetone.py](https://github.com/Fra0013To/deltaNN/blob/main/example_analyze_acetone.py) 
in this repository.
Anything related to these scripts (required data, results, etc.) are stored in the folder 
[*acetone_example*](https://github.com/Fra0013To/deltaNN/blob/main/acetone_example/) of the repository.

To run the examples (bash terminal):
1. Clone the repository:
    ```bash 
    git clone https://github.com/Fra0013To/deltaNN.git
    ```
2. Install the [required python modules](https://github.com/Fra0013To/deltaNN/blob/main/README.md#requirements).
    ```bash
    pip install matplotlib==3.4.2
    pip install numpy==1.19.5
    pip install pandas==1.2.4
    pip install PyYAML==5.4.1
    pip install scikit-learn 0.24.2
    pip install scipy==1.6.3
    pip install tensorflow==2.4.1
    ```
3. Run the script [example_train_acetone.py](https://github.com/Fra0013To/deltaNN/blob/main/example_train_acetone.py) 
for the training example:
    ```bash
    python example_train_acetone.py
    ```
4. Run the script [example_analyze_acetone.py](https://github.com/Fra0013To/deltaNN/blob/main/example_analyze_acetone.py) 
for the training example:
    ```bash
    python example_analyze_acetone.py
    ```

## Citation
If you find discontinuous NNs useful in your research, please cite:
#### BibTeX
> @Article{deltaNNs_2022,  
> AUTHOR = {{Della Santa}, Francesco and Pieraccini, Sandra},  
> TITLE = {Discontinuous Neural Networks and Discontinuity Learning},  
> JOURNAL = {Journal of Computational and Applied Mathematics},  
> VOLUME = {},  
> YEAR = {2022},  
> NUMBER = {},  
> ARTICLE-NUMBER = {},  
> ISSN = {},  
> DOI = {}   
> NOTE = {to appear (accepted for publication)}
> }
#### RIS
> TY  - EJOU    
> AU  - Della Santa, Francesco  
> AU  - Pieraccini, Sandra  
> TI  - Discontinuous Neural Networks and Discontinuity Learning  
> T2  - Journal of Computational and Applied Mathematics   
> PY  - 2022   
> VL  -     
> IS  -     
> SN  -     
> KW  - Discontinuous Functions     
> KW  - Neural Networks     
> KW  - Deep Learning   
> KW  - Automatic Detection of Discontinuity Interface  
> DO  -   

## Update
- 2022.08.11: Repository creation.

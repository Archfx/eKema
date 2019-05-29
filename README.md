# eKema
Machine learning on android
# Neural network based Paddy cultivation Inspection
<b>Revisited Version</b>

This report Contain justification for selection of mathematical functions related to the model.

> 1.Upgrades done to the Model<br>
> 2.System Design<br>
> 3.Network architecture<br> 
> 4.Implementation<br>
> 5.Results of the Model <br>
> 6.Inference Model <br>

### 1.Upgrades done to the Model

Completely New set of Training and Testing Date were collected from the field (During the period when implementing the model paddy cultivations were not in a situation to collect data). They contain six different classes of paddy images which is shown in the System Design. New Training and Testing Set contains more than 4,000 images that were collected from different fields.

Due to an Error in the Code base, when generating Test set the images were taken from Train set directory. Therefore, after training the Model, It was tested using same training set, this caused giving accuracy as 100%. This has been fixed in the Revisited version of the Model.

To minimize the loss of image details the size of images that were used to train and test the Model was increased from 50px to 64px. (Ideal size is 500px or More, Limited  to 64px due to Available performance in Training computer)

 ### 2. System Design

This model can identify 6 differentiations in paddy cultivations.

<h4><center>Healthy Paddy plants 
<img src="images\DataSet\Healthy (1).jpg" alt="" title="Healthy"  style="width:248px;height:128px;" /></center></h4>
<h4><center>Plants with nutrient deficiency
<img src="images\DataSet\LowFertile (1).jpg" alt="Alt text that describes the graphic" title="Title text" style="width:248px;height:128px;" /></center></h4>
<h4><center>Plants with Weed type 1 
<img src="images\DataSet\WeedType1 (1).jpg" alt="Alt text that describes the graphic" title="Title text" style="width:248px;height:128px;"/></center></h4>
<h4><center> Plants with Weed type 2
<img src="images\DataSet\WeedType2 (1).jpg" alt="Alt text that describes the graphic" title="Title text" style="width:248px;height:128px;"/></center></h4>
<h4><center> Plants with Weed type 3
<img src="images\DataSet\WeedType3 (55).jpg" alt="Alt text that describes the graphic" title="Title text" style="width:248px;height:128px;"/></center></h4>
<h4><center> Plants with Weed type 4
<img src="images\DataSet\WeedType4 (1).jpg" alt="Alt text that describes the graphic" title="Title text" style="width:248px;height:128px;"/></center></h4>

User can input a image/ images which contains paddy, model will classify the image/ images with percentages of above classifications. 

 ### 3. Network architecture

Five-layer neural network, input layer 64x64x3= 12 288, output 6 (6 types which were mentioned  in the System Design)
Output labels uses one-hot encoding





| Layer                   |  Dimensions                |
|-------------------------|----------------------------|
| input layer-X           |  [TrainSet size, IMG_SIZExIMG_SIZEx3]           | 
| 1 layer                 |  W1[IMG_SIZExIMG_SIZEx3, 200] + b1[200]<br>Y1[TrainSet size, 200]            |
| 2 layer                 |  W2[200, 100] + b2[100]<br>Y2[TrainSet size, 200]            |
| 3 layer                 |  W3[100, 60]  + b3[60]<br> Y3[TrainSet size, 200]            
| 4 layer                 |  W4[60, 30]   + b4[30]<br>Y4[TrainSet size, 30]             |
| 5 layer                 |  W5[30, 6]   + b5[6]     |
|  One-hot encoded label  |  Y5[TrainSet size, 6]             |

Layer implementation can be found [here](#layerimp)

#### model interpretation

<img src="images\Report\Equations\sm.png" alt="Alt text that describes the graphic" title="Title text" />


#### Why 5 -layer Neural Network

 This application is based on visual classification of images. The Differences between classes are due to small changes of features in images. In neural networks when we have more hidden layers the model can inspect more features that are available in the images when classifying. To get the maximum usage out of this the image sizes should be reasonable, and they should not have loosened their features upon resizing or manipulation. Therefore the Selection of 5 layers for this model was decided considering, taking the maximum usage of available hardware resources for better inspection  of features in the images.

This Model can be easily expandable for more classes and more Layers in future applications.

> ### Label representation
> Here are examples for each number, how each label is represented. These are the original pictures,
<img src="images\Report\paddy.png" alt="Alt text that describes the graphic" title="Title text" />

> Generation of data set according to the above Lable representaion can be found [here](#data)

> ### One Hot Matrix (One hot encoding)
> In deep learning we use y vector with numbers ranging from 0 to C-1, where C is the number of classes. If C is for example 4, then we have the following y vector which we will need to convert as follows:
<img src="images\Report\onehot.png" alt="Alt text that describes the graphic" title="Title text" />
In this model One Hot matrix technique is used for the encodeing of data.


### Selection of Error calculation function (Cost function) 

Cost functions calculate the difference between actual value (a) and the expected value (E). 
#### Quadratic cost
   Also known as mean squared error (Error sum of squares), maximum likelihood, and sum squared error, this is defined as:


<img src="images\Report\Equations\mst.png" alt="Alt text that describes the graphic" title="Title text" />
   The gradient of this cost function with respect to the output of a neural network and some sample r is:


<img src="images\Report\Equations\CaMst.png" alt="Alt text that describes the graphic" title="Title text"/>

   TensorFlow implementation- `tf.losses.mean_squared_error`<br>
 
 <center><b>Accuracy with Quadratic cost</b><br>
Train Accuracy: 0.9433127<br>
Test Accuracy: 0.802627<br>
F1 Score: 0.8673017086<br>
Time elapsed:5576.826694250107 Seconds<img src="images\report\qc.png" alt="Alt text that describes the graphic" title="Title text" style="width:350px;height:250px;"/></center>


#### Cross-entropy cost
   Also known as Bernoulli negative log-likelihood and Binary Cross-Entropy


<img src="images\Report\Equations\cce1.png" alt="Alt text that describes the graphic" title="Title text" />
   The gradient of this cost function with respect to the output of a neural network and some sample r is:


<img src="images\Report\Equations\cce.png" alt="Alt text that describes the graphic" title="Title text" />
   TensorFlow implementation- `tf.nn.sigmoid_cross_entropy_with_logits`<br>


<center><b>Accuracy with Cross-entropy cost</b><br>
    Train Accuracy: 0.9854759<br>
Test Accuracy: 0.81040895<br>
F1 Score: 0.8946081757097<br>    
Time elapsed:6233.307367324829 Seconds<img src="images\report\cce.png" alt="Alt text that describes the graphic" title="Title text" style="width:350px;height:250px;"/></center>



#### Exponentional cost
   This cost function requires choosing some parameter τ that we think will give us the behavior we need. We have to try different numbers until we get a better result. When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate.


<img src="images\Report\Equations\Exp.png" alt="Alt text that describes the graphic" title="Title text" />

This function can be used to calculate optimum learning rate for the model.

   TensorFlow implementation-`tf.train.exponential_decay`<br>
 <center> <b> This feature is expected to implement in future</b></center>


<h4> Selection of Cost function was done using Single number evaluation Metric, for this F1 Score is Used. This is a kind of Averaging method known as Harmonic Mean</h4>


<img src="images\Report\Equations\f1.png" alt="Alt text that describes the graphic" title="Title text" />

<center><h4>Since the best F1 Score of 0.8946081757097 is recorded from Cross-entropy cost, it was selected as Cost function</h4></center>


Cost function implementation can be found [here](#cost)


### Selection of Activation function
##### Linear Function
 <img src="images\report\linear.png" alt="Alt text that describes the graphic" title="Title text" style="width:250px;height:250px;"/>

 <img src="images\Report\Equations\lin.png" alt="Alt text that describes the graphic" title="Title text"/>
##### Sigmoid
<img src="images\report\sigmoid.png" alt="Alt text that describes the graphic" title="Title text" style="width:250px;height:250px;"/>


 <img src="images\Report\Equations\sigmoid.png" alt="Alt text that describes the graphic" title="Title text"/>
##### Tanh
 <img src="images\report\tanh1.png" alt="Alt text that describes the graphic" title="Title text" style="width:250px;height:250px;"/>

 <img src="images\Report\Equations\tanh.png" alt="Alt text that describes the graphic" title="Title text" />
##### ReLU
 <img src="images\report\relu.png" alt="Alt text that describes the graphic" title="Title text" style="width:250px;height:250px;"/>

 <img src="images\Report\Equations\relu.png" alt="Alt text that describes the graphic" title="Title text"/>
##### Leaky ReLU
 <img src="images\report\leaky-relu.png" alt="Alt text that describes the graphic" title="Title text" style="width:250px;height:250px;"/>


 <img src="images\Report\Equations\leakyR.png" alt="Alt text that describes the graphic" title="Title text"/>
##### Softmax

 Softmax function is a more generalized version of Sigmoid function. Since Sigmoid gives better results for classifiers and can only be used for binary classifiers Softmax function was created. This is a very popular Activation function among data scientists. The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs. This essentially gives the probability of the input being in a class. It can be defined as :

 <img src="images\Report\Equations\softmax.png" alt="Alt text that describes the graphic" title="Title text" />

> #### Selected Activation functions and Why

> Activation functions were selected based on logic and heuristics about them since there are no any rule of thumb to select them.

> - Sigmoid functions and their combinations generally work better in the case of classifiers but statndered Sigmoid function is used for binary classifiers. Softmax is more generalized type of Sigmoid function where we can use them in multi class classifiers
> - Sigmoids and tanh functions are avoided due to the vanishing gradient problem
> - ReLU function is a general activation function and gives best results.
> - But ReLU function work their best only in hidden layers.

>  This project initially started with ReLU function and tested for others. Considering the F1 score, activation function combinations were selected as Follows (All combinations were tested under Cross-entropy cost):

| Layer                   |  Activation Function                 |
|-------------------------|----------------------------|
| 1 layer                 |  ReLU |
| 2 layer                 |  ReLU     |
| 3 layer                 | ReLU      |
| 4 layer                 |  ReLU    |
| 5 layer                 |  SoftMax     |

<center><b>Accuracy with above Combination </b><br>
    Train Accuracy: 0.9854759<br>
Test Accuracy: 0.81908303<br>
        F1 Score: 0.8946081757097 <br></center>

 Previous Activation function Combinations

| Layer                   |  Activation Function                 |
|-------------------------|----------------------------|
| 1 layer                 |  ReLU |
| 2 layer                 |  Linear     |
| 3 layer                 | ReLU      |
| 4 layer                 | Linear  |
| 5 layer                 |  SoftMax     |

<center><b>Accuracy with above Combination </b><br>
    Train Accuracy: 0.9349215<br>
Test Accuracy: 0.78050196<br>
        F1 Score: 0.8507614361 <br></center>


| Layer                   |  Activation Function                 |
|-------------------------|----------------------------|
| 1 layer                 |  Linear |
| 2 layer                 |  Linear     |
| 3 layer                 | Linear      |
| 4 layer                 | Linear  |
| 5 layer                 |  SoftMax     |

<center><b>Accuracy with above Combination </b><br>
    Train Accuracy: 0.81223983<br>
Test Accuracy: 0.71080005<br>
        F1 Score: 0.758141818<br></center>




### 4. Implementation

> #### Available tools to implement Neural networks
> -  **MathLab**
-  **Python**
>  -  Using numpy (Math library for Python)
>  -  Using TensorFlow (Developed by Google)

> From above selection Python was selected due to the availability of Python. This model should be able to run on SOC devices (System on Chip devices). Implementing the model using python helps to make changes easily to run on different platforms.

> This Model was initially developed with only using numpy and the source code is available commented on this Report.

> The Main reason to move on to the TensorFlow is it has Cuda implementation where the model can be run on the GPU much faster than the CPU. TensorFlow has lot of built in features that supports for Neural networks

### 5.Results of the Model

> Results of the Model for Custom images can be found from [here](#results)

### 6.Inference Model
> Inference is used for Save the Parameters. By Saving trained parameters we do not need to run the program each time when we want to use the model.
> Using Inference Model an Android app was created.
> Inference of the Model can be found from [here](#inference)

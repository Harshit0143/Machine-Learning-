# Machine Learning 
Giving the ability to computers to perform tasks without explicitly being programmed 

### Supervised Learning
```Regression``` + ```Classification``` 
* We are given well labelles ```training data``` 
### Unsupervised Learning 
* ```Only input``` data is given. Machine has to find ```patterns```.   


    

## Terminologies 
* $y^{(i)}$ : ```output``` / ```target``` variable. It's a scalar value.   
* $x^{(i)}$ :  ```input variable``` / ```features```. It's a vector.  
* $(x^{(i)},y^{(i)})$ : ```training example```    
* $\\{(x^{(i)},y^{(i)});i = 1,2....n \\}$ : ```training set```.   
* ${h:X \mapsto Y}$ so that $h(x)$ is a good predictor is called the ```hypothesis``` where $X$ and $Y$ are respectively the spaces of ```input``` and * * * ```output``` variables. 
* ```Regression``` Problem: $Y$ is a ```continuous``` set.  
* ```Classification``` Problem: $Y$ is a set of small number of ```Discrete``` values.   


 # Linear Regression 
- We define $x_0=1$ which is an ```intercept term```. The ```hypothesis``` for ```linear regression``` is given as: Note that $\theta$ and $x$ are vectors.   
<p align="center">
<img width="200" alt="Screenshot 2023-04-04 at 11 37 26 AM" src="https://user-images.githubusercontent.com/97736991/229702344-f4cac09c-277e-45a5-a59b-5c756a7ef1eb.png"> </p>

- The ```cost function``` is given as:  
 <p align="center">
<img width="200" alt="Screenshot 2023-04-04 at 11 48 21 AM" src="https://user-images.githubusercontent.com/97736991/229704522-de7a4885-20bb-4149-8cf5-0357f43edf0a.png"></p>

#### ```Least Mean Square(LMS)``` Algorithm (Minimising the ```Cost Function``` / ```Batch Gradient Descent``` Algoeithm.  

1) Start with any ```guess``` for $\alpha$.  
2) Repeat until convergence: For each ```j```, ```SIMULTANEOUSLY``` perform the update for **$j = 0, 1, 2, 3.....d$** where **$d$** js the number of features. 

 <p align="center">
<img width="200" alt="Screenshot 2023-04-04 at 11 58 17 AM" src="https://user-images.githubusercontent.com/97736991/229706520-7d4618b6-f503-4189-8442-c6ee5ab95c92.png"> </p>

* $\alpha$: ```Learning Rate```. ```Small``` **$\alpha$** will require ```large``` number of iterations to converge. ```Large``` **$\alpha$** leads to
`divergence`.
* This simplifies to the `LMS update Rule` or the `Widrow-Hoff learning rule`.  
<p align="center">
<img width="600" alt="Screenshot 2023-04-04 at 12 34 52 PM" src="https://user-images.githubusercontent.com/97736991/229714199-ec648b39-9311-4344-ab62-ee7d11c7bc24.png">
</p>.  

* NOTE: ```Cost function``` for `Linear Regression` is a `Convex Function` so it has `EXACTLY one minimum`. Given that **$\alpha$** is not too large, the `Gradient Descent` always converges to the `ONLY` minima. Irrespective of `Initial Guess`.   
* The contour plot:  Points on the same ellips have the same `Cost`. As discussed, the ellipse with the `Optimal Cost` degenerates to a `point`.
<p align="center">
<img width="585" alt="Screenshot 2023-04-04 at 12 39 17 PM" src="https://user-images.githubusercontent.com/97736991/229715190-6a1a9a3d-587a-481b-adbd-80682d426727.png"></p>

#### ```Stochastic Gradient Descent```/ `Incremental Gradient Descent`.  
<p align="center">
<img width="600" alt="Screenshot 2023-04-04 at 1 08 35 PM" src="https://user-images.githubusercontent.com/97736991/229721806-52ca0cd8-fbb7-445e-8893-557c15553279.png"> </p>

* So we are ```periodically``` updating w.r.t training example $i = 1, 2, 3.....n$ 
* While in `Batch Gradient descent` we would do it with the sum at each step. 
* This method is `faster` but ```does not converge``` but It will keep oscillating aroound the ```final value``` because:
* it is something like: You keep `periodically` changing the `training set` (which is a singleton) at each step.
*  This though will be a very good approximation and practically there is `Terrabytes` of training data so it won't be possible to read the `entire training data` in each iteration which is needed in the `Batch Gradient Decent`. 
*  So the ```Stochastic Gradient Descent``` takes a `noisy` path averages towards the `Global Minimum`. 
*  By slowly letting the learning rate $\alpha$ decrease to zero as the algorithm runs, it is also
possible to ensure that the parameters will converge to the global minimum rather than
merely oscillate around the minimum.

*  More compactly, the update is. Note that $x^{(i)}$ is a vector:
 <p align="center"><img width="400" alt="Screenshot 2023-04-04 at 1 18 41 PM" src="https://user-images.githubusercontent.com/97736991/229724291-bb5dc757-4238-4544-8ff5-2ba5dcf68560.png">
</p>

## Implicit Expression for `Global Minimum` of `Linear Regression`. 

## Matrix Derivatives  

<p align="center">
<img width="1000" alt="Screenshot 2023-04-04 at 1 32 59 PM" src="https://user-images.githubusercontent.com/97736991/229727707-db7c0418-c840-4ba8-9180-b6c4a4e6ed46.png"> </p>

## Define
<p float="center">
  <img width="375" alt="Screenshot 2023-04-04 at 1 39 48 PM" src="https://user-images.githubusercontent.com/97736991/229729376-247f6874-ab5e-4d89-9513-f6e14beeb31b.png"> 
 
  <img width="225" alt="Screenshot 2023-04-04 at 1 40 10 PM" src="https://user-images.githubusercontent.com/97736991/229729456-8acec260-2963-41e2-bc93-4dadfa738ad8.png">
</p>

## Note that $X$ is a $n$ x $(d+1)$ matrix and $\vec{y}$ is a $n-dimensional$ column vector.   
 
 
 
and noting <br>
<img width="503" alt="Screenshot 2023-04-04 at 1 42 29 PM" src="https://user-images.githubusercontent.com/97736991/229730060-3ca8215a-25d2-41aa-bcbd-cc6a9fd39992.png">. 
We get 

<p align="center">
<img width="553" alt="Screenshot 2023-04-04 at 1 43 16 PM" src="https://user-images.githubusercontent.com/97736991/229730242-031aa59d-f62c-49f1-bddb-394c31b287a0.png"> </p>
Setting 
<img width="90" alt="Screenshot 2023-04-04 at 1 45 11 PM" src="https://user-images.githubusercontent.com/97736991/229730702-d3862ef5-29f8-427a-ab59-bd6a80925478.png">

to ```0```. 
We obtain the `Normal Equation`: 
<p align="center">
<img width="817" alt="Screenshot 2023-04-04 at 1 46 22 PM" src="https://user-images.githubusercontent.com/97736991/229730932-6a56d2da-4d3d-40fb-9c90-ef674623170f.png"></p>



# Locally Weighted Regression 


## Parametric Learning Algorithms 
* It is not necessary that the ```hypothesis``` is linear in the `features`
* How do we devise if the `hypothesis` should depeend on $sqrt(x)$ or $x^2$ or $log(x)$ 
* `Feature selection Algorithm` does that 
### Parametric Learning Algorithm (eg `Linear Regression`)
* Fit fixed set of Parameters ($\theta _i 's$ in linear regression)
### Non-Parametric Learning Algorithm (eg `Locally Weighted Regression`) 
* Amount of `Data/ Parametes` stored grows (here linearly) with size of `training set`
* So it is not very suitable for a very large `training set`

Algorithm 
<p align="center">
<img width="800" alt="Screenshot 2023-04-04 at 2 59 34 PM" src="https://user-images.githubusercontent.com/97736991/229749603-c14cb29e-9291-4638-bdc3-0b2c0b00d720.png"></p>

<p align="center"><img width="400" alt="Screenshot 2023-04-04 at 3 01 53 PM" src="https://user-images.githubusercontent.com/97736991/229750228-a566125e-a71a-412e-9020-1c58dff3839e.png">
</p>

 

#### here $\tau$ is a chosen `appropriately` {How much we want $w^{(i)}$ to decay with distance from $x$}
#### value of $\tau$ has an effect on overfitting and underfitting the data. 

* So bascallly, we are giving more `weight` to the points `close` to the $x$ where we want to predict
* Compute the weights $w^{(i)}$ at the desired $x$, then get the $\theta _i's$ using any of the prvious Algorithms. Then get $h(x)$


# Probabilistic Interpretation to `Linear Regression`
#### Why use `Squared Error` in the `cost expression`? 
# $$y^{(i)} = \theta ^T x^{(i)} + \epsilon^{(i)}$$ 
* where $\epsilon^{(i)}$ is an `error` term carrying `unmodelled` effects like `noise`. $\epsilon^{(i)}$ is distributed `Normally` with mean `0` and variance $\sigma ^2$. 

* $\epsilon^{(i)} ; i = 1, 2,3 .....n$ are assumed `i.i.d.` which is not vary valid (for example, if due to some reason, the housing price on a street is high, it will make the prices of the other houses on the street also high.  
# LEFT FOR ANOTHER DAY  


# Logistic Regression (Classification)

## Binary Classification
### Terminologies 
* Binary Classification ==> $y ∈ \\{0, 1\\}$
* $1$ or `Positive Class` or $+$
* $0$ or `Negative Class` or $-$ 
* given `training example` $(x^{(i)},y^{(i)})$ , $y^{(i)}$ is called the `label` of $x^{(i)}$  
Clearly, using `Linear Regression` here is pointless, $0$ and $1$ are `labels` to represent classes, ther values doesn't concern us. 

We use the `sigmoid function`/ `logistic function`. 
<p align="center">
<img width="484" alt="Screenshot 2023-04-04 at 3 51 44 PM" src="https://user-images.githubusercontent.com/97736991/229762923-c0071761-f8d9-4d5c-874c-25241a2fb350.png"> </p> 
We interpret the output as:  
<p align="center">
<img width="400" alt="Screenshot 2023-04-04 at 3 54 05 PM" src="https://user-images.githubusercontent.com/97736991/229763491-cd90915c-ec2b-4576-a1a8-e94f43494813.png"></p>


* So we will finally use `Gradient Descent` to maximise the `log(likelyhood)`.$l(\theta)$ is a `conave function` so there is `EXACTLY` one maxima.
<img width="966" alt="Screenshot 2023-04-04 at 4 10 38 PM" src="https://user-images.githubusercontent.com/97736991/229767433-ce2973b3-b909-4e4e-8253-654769df64e0.png">
* The update rule turns out to be same as the `Linear Regressoin`, only that it least to `ascent` this time.
<p align="center">
<img width="1153" alt="Screenshot 2023-04-04 at 4 17 59 PM" src="https://user-images.githubusercontent.com/97736991/229769160-4ce89d9c-1dc2-487e-9455-1703f7cd82c6.png"></p>


# Newton's Method (faster method to obtain optimum $\theta$ in logistic regression)
* It's the same we learnt in MTP290 to find extrema ( extremum of $f(\theta )$ <==> root of $f'( \theta)$ for `concave/ convex` function). The method showed `quadratic convergence` given that multiplicity of root was `1`. 
* There is some Math when $\theta $ i.e. the `function input` is multidimensional. 
* The takeaway is that the `time` for iteration grows with the number of parameters. ($H$ is a $(d+1)$ x $(d+1)$ matrix. Each iteration takes atleast $O(d^2)$ time)
* One iteration of `Newton Raphson` is costlier than one iteration of `Batch Gradient Descent`. If there are just `10-15 features`, it is beneficial to use `Newton Raphson`.


<p align="center">
<img width="995" alt="Screenshot 2023-04-04 at 4 26 03 PM" src="https://user-images.githubusercontent.com/97736991/229770974-c7b812b9-c765-4d83-87ea-a410c5941bdc.png"></p>



# Perceptron Algorithm 

    g(z) = 1 if z >= 0
           0 otherwise 
    

* The update rule is still the same.   
<p align="center">
<img width="600" alt="Screenshot 2023-04-04 at 4 56 28 PM" src="https://user-images.githubusercontent.com/97736991/229777534-8aaa3b86-4732-460b-a7ac-4834a4ba8100.png"></p>

* Go over, example by example in an online manner (use the `stochastic descent` update rule) and keep updating the `decision boundary`

$y^{(i)}-h_\theta (x^{(i)})= $   
 &ensp; 0   if algorithm get's it right.  
 &ensp; 1   if $y^{(i)} = 1$ and $h_\theta (x^{(i)})= 0$      
-1   if $y^{(i)} = 0$ and $h_\theta (x^{(i)})= 1$

<p align="center">
<img width="768" alt="Screenshot 2023-04-04 at 5 32 34 PM" src="https://user-images.githubusercontent.com/97736991/229785556-448913b3-9d5d-49ba-bd95-f7792764411f.png"></p>

The mechanism:   
1) We intially have a vector $\theta$ and the corresponding `decision boundary`. (they will be normal to each other).  
2) We add a new `training example` to the `training set`.  
3) If the predicted output $h_\theta (x^{(i)})$ == the actial output $y^{(i)}$ then we don't make any change.  
4) Otherwise, $y^{(i)}-h_\theta (x^{(i)})$ will be either `1` or `0` and out `update rule` will `tilt` $\theta$ , hence the `decision boundary` in the requisite direction (try running it on the example in the picture).  
Geometrically:  
* We add a component of $x$, specifically, $\alpha x$ to $\theta$ resulting in $\theta'$. This `tilts` $\theta$ in the `required direction` 
  
<p align="center">
<img width="257" alt="Screenshot 2023-04-04 at 5 41 29 PM" src="https://user-images.githubusercontent.com/97736991/229787375-5f930efd-bf70-419b-b6f4-ea747482cd9a.png"></p>


* Such a strict decision boundary will fail in classifying something like this.
<p align="center">
<img width="162" alt="Screenshot 2023-04-04 at 5 44 13 PM" src="https://user-images.githubusercontent.com/97736991/229787958-ef186f1b-f2ab-4fcb-a37f-b2f262968003.png"> </p>

* In the `Perceptron Algorithm`, we essentially demand that all points of `label 1` are on `one side` of the boundary, and all points of `label 0` are on other side of the boundary. This can't accomodate `noisy data` i.e. some not very well fitting exaples.


# Exponential Family

<p align="center">
<img width="491" alt="Screenshot 2023-04-04 at 5 54 18 PM" src="https://user-images.githubusercontent.com/97736991/229790559-57071c65-9077-44ba-87b4-2869e7f6079a.png"> </p>
* The above should integrate to `1` for a valid choics of the following parameters. 
* y : Data (scalar)
* η : `Natural Parameter`/ `Canonical parameter`  (vector) 
* $T(y)$ : Sufficient statistic  (vector with same dimension as η)
* a(η) : Log partition function (scalar)
* b(η) : Base Measure  (scalar)

#### We will try to `manipulate` the `pdf` given ti us into the `above` form. 
Example: 
### Bernoulli Distribution**  

<p align="center">
<img width="715" alt="Screenshot 2023-04-04 at 6 08 18 PM" src="https://user-images.githubusercontent.com/97736991/229793962-ed99c251-73b5-47d4-b8ba-60ffac70baa3.png"></p> 

which leads to: 
<p>
<img width="300" alt="Screenshot 2023-04-04 at 6 09 39 PM" src="https://user-images.githubusercontent.com/97736991/229794306-0721fe59-a87f-4602-ab19-7ff860403f79.png">
    &ensp;&ensp;&ensp;&ensp;&ensp;

<img width="80" alt="Screenshot 2023-04-04 at 6 10 29 PM" src="https://user-images.githubusercontent.com/97736991/229794527-02ebc4a3-6d88-43e5-9e97-5301d0460674.png">
<img width="168" alt="Screenshot 2023-04-04 at 6 10 19 PM" src="https://user-images.githubusercontent.com/97736991/229794480-388bb73f-a3ad-4233-8179-64914501c851.png">
</p>
<img width="337" alt="Screenshot 2023-04-04 at 6 09 53 PM" src="https://user-images.githubusercontent.com/97736991/229794367-5be8373a-394f-4840-8c8d-a93eea5b7ac3.png">




So it verifies that the `Bernoulli Distribution` is a member of `Exponential Familiy` (as we are able to solve for all the parameters explicitly)

### Gaussian distribution

<p align="center">
<img width="918" alt="Screenshot 2023-04-04 at 6 15 36 PM" src="https://user-images.githubusercontent.com/97736991/229795803-b68cdeac-e4f5-4bc1-8699-4cb4325c8391.png"> </p>




* Gaussians with any variance $σ^2$ beling to the `Exponential Family`. It has been skipped here.  

### Properties if Exponential Family
* `MLE` (`Maximum Likelyhood`) w.r.t. η is `concave`. 
* `NLL` (`Negattive Log Likelyhoood`) is  `convex` (it's a minimisation problem)
* $E[y;η] = \frac{\partial &ensp;a(η)}{\partial η}$
* $Var[y,η] = \frac{\partial^2 &ensp;a(η)}{\partial η^2}$

    Real values Data: Gaussian
    Binary Data: Bernoulli 
    Count(non-negative integers): Poisson 
    Positive Reals: Gamma/ Exponential 
    pdf over pdf: Beta, Dirichlet } Part of Bayesion Statistics 

# Generalised Linear Models (GLM's)
### Assumptions/ Design Choices 
1) $y|x;\theta$ ~ Exponential(η)
2) $η = \theta^Tx$ where $\theta,x \in \Re^n$, where $n$ is the dimension of the input
3) Test time: $h_\theta(x) = E[y|x;\theta]$ 




<p align="center">
<img width="828" alt="Screenshot 2023-04-04 at 6 44 18 PM" src="https://user-images.githubusercontent.com/97736991/229803038-63c69781-645a-4f32-94ab-fb13a34330c0.png"></p>

* Note that we are `learning` $\theta$.  
* In the bigger picture, we get a scalar $η = \theta^Tx$ from the input $x$, vector. Then we put it in a suitable `p.d.f.` to get the output (The `hypothesos value`). This is basically the `GLM`. 

## Training the `GLM` 
* Good News: Whatever `funciton` we take from the `exponential family`, the `Learning Rule` is the same:     
For `Batch Gradient Descent`: 
<p align="center">
<img width="804" alt="Screenshot 2023-04-04 at 6 49 45 PM" src="https://user-images.githubusercontent.com/97736991/229805113-e531ebef-6575-4e8f-84fb-e08b0ab25419.png"></p>


For `Stochastic Gradient Descent`:  
<p align="center">
<img width="862" alt="Screenshot 2023-04-04 at 6 50 26 PM" src="https://user-images.githubusercontent.com/97736991/229805442-ac449520-35a6-447e-9b0c-d5b7fe42e5c1.png"></p>

* We have generalised, `Regression`, `Classification` and everything into one class. 



* $\mu = E[y;η] = \frac{\partial &ensp;a(η)}{\partial η} = g(η)$ = `Canonical Response function` 
* $η = g^{-1}(\mu)$ : `Canonical Link function` 

### 3 parameters 
1) Model Parameter: $\theta$
2) Natural Parameter: $\eta$
3) Canonical parameters: Bernoulli($\phi$), Gaussian($\mu,\sigma^2$), Poisson($\lambda$).       
Note:       
       $\theta^Tx \to \eta$  (1 --> 2).   
       $g(\eta) \to \mu$    (2 --> 3). Note that the `mean` in poisson equals $\lambda$ and that in Bernoulli equals $\phi$  
       $g^{-1}(\mu) \to \eta$  (3 --> 2).          


* `Logistic` regression is just using Bernoulli 
* If we want to predict the number of visitors on a website, we use `Poisson`

* For the `Gaussian`, we assume `variance = 1`
<p align="center">
<img width="794" alt="Screenshot 2023-04-04 at 7 24 51 PM" src="https://user-images.githubusercontent.com/97736991/229815343-c1f73e53-df54-4d24-8909-131799d4fa13.png"></p>


<p align="center">
<img width="606" alt="Screenshot 2023-04-04 at 7 26 13 PM" src="https://user-images.githubusercontent.com/97736991/229815786-ce88e68b-154e-4005-b581-1b9c1445d5c2.png">></p>

## Softmax Regression (non-GLM approach)
* Extending the idea of `Binary Classification`, we wish to classify into `n categories` {Say Dog, Cat, Hanster...}
* Cross Entropy 
<p align="center">
<img width="443" alt="Screenshot 2023-04-04 at 7 28 21 PM" src="https://user-images.githubusercontent.com/97736991/229816382-b38bb793-dad2-4d51-9bb1-b7c0e6662122.png"></p>


* $k$ classes 
* $x^{(i)} \in \Re^n$ 
* $\theta_{class} \in \Re^n$
* $class \in \\{ object_1, object_2......opject_k\\}$
* The `featurs` of all the classes are same. The `weights` are different i.e. $S_i = \\{\theta_j ; j = 1, 2......d\\}$ for $i \in {1, 2....k}$ the $S_i's$ are different. 
* So for each class $class_i$ we have the `Decisioin Boundary` $\theta_i^Tx = 0$, the boundary of `in class_i` and `not in class_i`
* The below is a very `beautiful/ ideal` exapmple where there is no overlapping. 


<p align="center">
<img width="533" alt="Screenshot 2023-04-04 at 9 37 51 PM" src="https://user-images.githubusercontent.com/97736991/229852108-0816a253-41da-4e44-98d7-a6eae3ee59c9.png"> </p>

* Logit Space
* Choose the $x$ you want to predict.
* Calculate $\theta_i^Tx$ for each $i = 1, 2, 3.....k$
* Exponenentiate the values 
* Normalise them (so that sum of falues for  $i = 1, 2, 3.....k$ is 1
* You have the probabilities. $P_i$ = Probability that the predicted `point` lies in $class_i$ 
* Define $\hat{p}(y)$ as the predicted value


<p align="center">
<img width="946" alt="Screenshot 2023-04-04 at 9 41 14 PM" src="https://user-images.githubusercontent.com/97736991/229852902-6f74995b-d61a-4cb9-a7c7-3f5249640a96.png"></p>

### How to learn? We still don't know how to get $\theta_i$ 
* We try to minimise the `Cross entropy` between $p(y)$ and $\hat{p}(y)$. 


<p align="center">
<img width="707" alt="Screenshot 2023-04-04 at 9 52 37 PM" src="https://user-images.githubusercontent.com/97736991/229855612-2e796495-85ab-48ec-98f7-dab9ec1e9a27.png">></p>

* Use `Gradient Descent` now   

## GLA (`Generative Learning Algorithm`)
* In `Logistic Regression`, we try to make a `linear` `decision boundary` using the training set.
* In GLA, we consider each `label`, `separately`, make a `model`, then then we have to make prediction on $x$, we see which model it matches to.
* It is simpler and sometimes computationally more efficient. 

### `Discriminative Learning Algorithm` learns (Logistic Regression), We `discriminate` between the different `classes`
* $P(y|x)$ ; Note that $x$ is a vecțor and $y$ is a scalar. Given `training` values **$x^{(i)}$** we are `learning` $y^{(i)}.  
* $h_\theta(x) =  0$ or $1$ 
### Generative Learning Algorithm learns:
* $P(x|y)$  ; i.e. given the `label`, we are `learning` the `featires` associated with it 
* $P(y)$ i.e. `Class Prior`
* So it's statistics. 
1) The patient walks in
2) You `test` whether tumour is malignant. This data over many patients gives $P(y)$. 
3) Now you note down the `features` on the specific patient and map it to malignant or not malignant. This gives $P(x|y)$. 
 


<img width="1061" alt="Screenshot 2023-04-05 at 10 26 50 AM" src="https://user-images.githubusercontent.com/97736991/229984743-45235c0b-78c3-464c-b76b-51771a468f63.png">

* Then we can use Baye's Rule
* $P(x|y = 0)$ and P(x|y = 1)$  are learnt in `GLA`
* $P(y = 0)$ and $P(y = 1)$  are learnt in `GLA`
* Get $P(x)$ using:
$$P(x) = P(x|y=1)P(y=1) + P(x|y=0)P(y=0)$$
* Now get $P(y=1|x)$, using.  
$$P(y=1|x) = \frac{P(x|y = 1)P(y=1)}{P(x)}$$


* Generative Algorithms 
1) Continuous Features (predicting tumour label)
2) Discrete Features (predicting spam email)


## Gaussian Discriminal Analysis 
* Suppoes $x\in \Re^d$, we are dropping the $x_0 = 1$. 
* `Assume` $P(x|y)$ is Gaussian. Note that $x$ is a vector.  

## Multivariate Gaussian 
<p align="center">
<img width="835" alt="Screenshot 2023-04-05 at 10 50 15 AM" src="https://user-images.githubusercontent.com/97736991/229987830-fad78d34-418a-47e4-9c62-a0ef8906c9e5.png"> </p>

* $d$: number of features.   
* µ ∈ $\Re^d$: `mean vector`.  
* Σ ∈ $\Re^{d&thinsp;Χ&thinsp;d}$ : `Covariance Matrix`. It is `symmetric` 


<img width="628" alt="Screenshot 2023-04-05 at 10 57 16 AM" src="https://user-images.githubusercontent.com/97736991/229988800-da492ba4-f151-495e-9555-54e3f2d8990c.png">
<p align="center">
$Cov(Z) = E[(Z − E[Z])(Z − E[Z])^T] = E[ZZ^T]−(E[Z])(E[Z])^T$. </p>


#### Left to right show `Gaussian Curves` each with `mean = 0 vector` and `covariance matrix` respectively $I$, $2I$, $0.6I$.  
<p align="center">
<img width="1035" alt="Screenshot 2023-04-05 at 11 00 18 AM" src="https://user-images.githubusercontent.com/97736991/229989256-00ac9605-80ef-4ba8-86d7-0e45c756bce3.png"></p>

* As Σ becomes `larger`, the `Gaussian` `spreads out`
* As Σ becomes `smaller`, the `Gaussian` `compresses`



<p align="center">

<img width="650" alt="Screenshot 2023-04-05 at 11 32 56 AM" src="https://user-images.githubusercontent.com/97736991/229994363-c7d22e2f-cfc3-4c41-9bdb-61df3b293d34.png">

<img width="911" alt="Screenshot 2023-04-05 at 11 34 57 AM" src="https://user-images.githubusercontent.com/97736991/229994702-651391f4-7d78-4435-baee-b2b1cd7dd9d6.png">
</p>


# Setting the non-diagonal Elments negative: 

<p align="center">

<img width="992" alt="Screenshot 2023-04-05 at 11 31 29 AM" src="https://user-images.githubusercontent.com/97736991/229994111-83adf09e-7763-4c30-a851-d8f20cbd7add.png"> 
</p>


## `Gaussian Discriminal Analysis` (`GDA`) model 
#### The `GDA` model iș $P(x|y)$ is modellled as a `multivariate` Normal Distribution. Note that the `mean` for $P(x|y=0)$ and $P(x|y=1)$ are different but the `covariance matrix` for both is the `same`

<p align="center">
<img width="820" alt="Screenshot 2023-04-05 at 11 59 05 AM" src="https://user-images.githubusercontent.com/97736991/229998869-bbc70d86-3972-4d8f-9c75-61be8b438724.png"></p>

## Now our aim is to fit $\mu_1$, $\mu_2$, $Σ$ and $\phi$ into a the `training set`. 
## Fitting here, is defined as `maximising` the `Joint Likelyhood`. We will maximise the `log likelihood` (the `log` of `Joint Likelyhood`)
<p align="center">
<img width="600" alt="Screenshot 2023-04-05 at 12 14 59 PM" src="https://user-images.githubusercontent.com/97736991/230001901-f026b435-0a22-41aa-8eab-fa9f6d104d0e.png"></p>

#### On doing the `math` we find the `optimal` parameters to be. Here $1\\{\\}$ is the `indicator function`.  

<p align="center">
<img width="597" alt="Screenshot 2023-04-05 at 12 16 03 PM" src="https://user-images.githubusercontent.com/97736991/230002128-3ba99f77-ae4e-488d-b2bb-b74faeee8b88.png"></p>

Note the difference. In `Logistic Regreession` we were trying to `maximise` the `Conditional Likelihood`: 
$$L(\theta)= \prod_{1}^{n}P(y^{(i)}| x^{(i)},\theta)$$


<p align="center"><img width="646" alt="Screenshot 2023-04-05 at 12 37 44 PM" src="https://user-images.githubusercontent.com/97736991/230006464-6685315b-4cde-4322-be82-11acba8baa70.png">
</p>

* The `GDA` tries to fit `2 Gaussians` over the training set. i.e. the `x` with same `predicted probability` lie on the same `contour`. (Note that the `prediction` is different from the $y^{(i)}$ given in the `traiining set` which is either `0` or `1`.
* The `rule` for fitting, we already discussed, masimising the `log likelihood`. 
* Note that the `covariance matrix` for both are same. So the `2` curves are `identical` in `shape` and `orientation` (The `decay rate` at same distance from `respective centres`) 
* The straight line is the `Decision Boundary` Drawn by logistic Regression
### It's actually pretty cool to see how `Logistic Regression` and `GDA` will report probabilites.
* In `Logostic Regression` we have a decision boundary. All lines `parallel` to it are contours. The `uncertaininty` is the maximum at the `Decision Boundary` and it reduced as the `Sigmoid Function` as you move away, `normal` to the `Decision Boundary`.   
* In `GDA`, the probability of identifying as a $class_0$ is maximum at the `peak` if the `Gaussian` of $class_0$ shown in the figure and decaus out as the `Gaussian` on moving away from the peak. The `peak` is supposed to be somewhat at the `average` of the cluster.
* The gaussian gives the `probaility` of it being that $x$ fot the given `class`. 
* So we can't compare the same $x$ from 2 different `Gaussians` directly. But if we multiply each `Gaussian K`  by $frac{P(y = k)}{P(x)}$ then we can compare the `Gaussians`. Note that they will still remain `Gaussians`, just `not normalised`




* The `predicted` `class` is given by: 
 $$\underset{y}{arg max } P(y|x) = \underset{y}{\arg max}  \frac{P(x|y)P(y)}{P(x)}$$
* `Note` that the $P(x)$ in `RHS` Denominator is constant for every $y$ so you can skip it to save `computation cost`. 
* So basically, for each `class` you find which `contour` yout $x$ lies on, get $P(x|y)$, then calculate the `probability` $x$ belongs to that class using `Baye's Rule`. Then you give the class which has the `maximum probability` as your `prediction`.  

#### These are the `decision boundaries` drawn by the 2 algorithms:  
<p align="center">
<img width="853" alt="Screenshot 2023-04-05 at 1 38 20 PM" src="https://user-images.githubusercontent.com/97736991/230020724-026d49cf-e7a8-48af-9a4f-d3754963cd77.png"></p>

* Using the `same covariance matrix` for both the `classes` ends up with `less` parameters to deal with and a `linear` decision boundary. 

## Comparison of `GDA` to `Logistic Regression`
* Set fixed $\mu_0, \mu_1, \Sigma, \phi$
* Assume $x$ is `1 dimensional` i.e. there is just 1 feature   
* We plot $P(y = 0|x)$ and $P(y = 1|x)$   {It's value from the previous expressoins we saw} vs $x$
* Take $P(y=0) = P(y=1) = 0.5$
* Then plot $max(\frac{P(y = 0 | x)P(y)=0}{P(x)},\frac{P(y = 1|x)P(y=1)}{P(x)})$  note which graph it 
* We find that the curve is `Exactly` the `sigmoid function`. 
* Varying the parammeters $\mu_0, \mu_1, \Sigma, \phi$ is effectively varying $\theta^T$ in `Logistic Regression`. 
<p align="center">
<img width="754" alt="Screenshot 2023-04-05 at 2 38 14 PM" src="https://user-images.githubusercontent.com/97736991/230035657-6630f1fd-e813-406d-bcbf-f6e026df4f0f.png"></p>

* If we make stringer `modelling assumptions` and the `assumptions` are reasonably `co  rrect`, the model will berform much better.
* The `GDA` makes a `Stronger set of Assumptions` and `Logistic Regrssion` makes a `weaker set of assumptions` {You cam prove `if` (stronger) `then` (weaker)}
* Any memeber from the `exponential family` follows this. 
* If we don't know whether our `data` is `Gaussian` or `Poisson`, using `Logistic Regression` will do fine (as it needs no assumptions)
* However, if we know our `data` is `Poisson` and we `feed` it into the `Algorithm`, (we are giving in `more` information), our `Algorithm` will do better.  
* `Weaker` assumptions means `Algorithm` is more `Robust` to modelling assumptions (performance less affected by what distribtion data folllows)
* `Algorithm` has `2` sources of knowledge:
     1) The data 
     2) The information (type of distribution) data follows, given by you
* In Practice: `Plot` the data (if it's large enough) and see which `statistical model` it follows. Then decide what you wish to use
* The skill in ML when you can male prediction with smaller data (100 traininig examples and not a million training examples). 


# `Naive Bayes`
* We have a `binary feature vector` $x \in \\{0,1\\}^d$ where $d$ is the `dictionary` size. 
* An `email` is converted to corresponding $x$ where $x_j = 1$ `if` $word_j$ is present in the email, `otherwise` $x_j = 0$. 
* The `feature vector` is called `vocabulary`. 
### We are building a `Generative Learning Algorithm`. 
* A very strong assumption (`Conditional Dependence Assumption` or `Naive Bayes Assumption`) is made. 
* We assume that $x_i’s$ are conditionally independent given y.
* i.e. given whether the email is `spam` or `not spam`, knowing that a word $word_i$ appears in the email does not affect the probability that another word $word_j$ appears in the email. This is not vey accurate. 
* This gives us: 
<p align="center">
<img width="853" alt="Screenshot 2023-04-05 at 3 20 59 PM" src="https://user-images.githubusercontent.com/97736991/230046225-28968167-9d58-421c-8ed2-b74522832ab9.png"></p>

### Parametes
* $y \in \\{0,1\\}$ and $y = 1$ <==> email is `spam` ̦
* $\phi_{j| y = 1 } = P(x_j = 1 | y = 1)$
* $\phi_{j| y = 0 } = P(x_j = 1 | y = 0)$
* $\phi_{y} = P(y = 1)$

### Fitting 
#### Maximise the joint likelihood: 
<p align="center">
<img width="519" alt="Screenshot 2023-04-05 at 3 27 49 PM" src="https://user-images.githubusercontent.com/97736991/230047869-01207ada-4f7f-4e58-b6d3-d800e2c84c0d.png"></p>

#### That leads to the `optimal` parameters: 
<p align="center">
<img width="546" alt="Screenshot 2023-04-05 at 3 28 45 PM" src="https://user-images.githubusercontent.com/97736991/230048128-23804954-d13d-4434-985e-3ae768085574.png"></p>

* Here $n$ is the number of training examples.  
* This matches intuition.
*  Optimal $\phi_{j| y = 0 }$ corresponds to `fraction` of `non-spam` emails that had the word `j`
*  Optimal $\phi_{j| y = 1 }$ corresponds to `fraction` of `spam` emails that had the word `j`

## How to detect a new spam email?
<p align="center">
<img width="1016" alt="Screenshot 2023-04-05 at 3 44 30 PM" src="https://user-images.githubusercontent.com/97736991/230051766-0cb79e88-c049-41a5-bec5-6cefb926dc75.png"></p>

#### `Regression` will perform better than this but this is `computationally` very efficient.

### `Laplace Smoothing`
* There are some issues with the `Naive Bayes algorithm`
* `Naive Bayes` is computationally cheap and easy to implement compared to `regression` 

## `Support Vector Machines`
* We aim to draw `non linear` boundaroes. 
* We need to mao input featires, match this to a `high dimensional` set of fretures:
* Then alply linear regressoin on the obtaind featires. 
* For example if we have features $x_1$ and $x2$, the output can depend on $\sqrt(x_1x_2)$, $x_1^2$, $log(x_2)$.  
* These are not as effective as `Neural Networks`
* But the implementation is much easier due to `packages`  

#### `Optimal margin Clasiifier` (Separable case)

<p align="center">
<img width="250" alt="Screenshot 2023-04-06 at 9 52 27 AM" src="https://user-images.githubusercontent.com/97736991/230271463-5564f747-e87e-4c13-91b0-cc456a15768b.png"></p>

* First we derive an algorithm for `linear boundary` case. {We'll look at the `Inseparable case` later} 

#### `Functional Margin`


### `Geometric Margin` (Recall Data is `linearly separable`) 
* We are trying to maximise the `Geometric Margin` 
<p align="center">
<img width="1095" alt="Screenshot 2023-04-06 at 10 09 18 AM" src="https://user-images.githubusercontent.com/97736991/230273411-b6ff89fe-077b-46cb-ae44-f5c01f933680.png"></p>

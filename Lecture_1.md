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



















































































































































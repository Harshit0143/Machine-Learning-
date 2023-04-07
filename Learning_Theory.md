# Learning Theory

#### Assumptions 
1) There exists a `Data Distribution` $D$ 
* $(x,y)$ ~ $D$ 
* We will both, `train` and `test` from $D4 
2) All `samples` are sampelled `independently` 


 <p align="center"><img width="1177" alt="Screenshot 2023-04-07 at 5 33 36 PM" src="https://user-images.githubusercontent.com/97736991/230605657-5a6ae884-a21c-457e-b117-e56945fd40b8.png">
</p>

* $S$ is a `Random Variable`
* The `Learning Algorithm` is a `Deterministic Function` 
* Hemce, $\hat{h}$ is a `Random Variable` 
* There exists `true` parameters $\theta^{\*}$ and $h^{\*}$ {Not `random`. It's a `constant`}
* $\(^{\*}\)$ denotes a `true` quantity 
* $\hat{x}$ denotes a `prediction` of $x$

## `Data View` 
 <p align="center">
<img width="652" alt="Screenshot 2023-04-07 at 5 49 46 PM" src="https://user-images.githubusercontent.com/97736991/230607766-36b16c71-10a9-442e-8554-bbac423e83a7.png"></p>

## `Parammeter View`
* The distribution $D$ keeps producing a `Training Set` of size $m$
* Each time we run our `4` algorithmns and get the value of $\theta$
* We plot it. here `feature vector`, hence $\theta$ is taken as a `2-dimensioanl`
* `star` corresponds to the `true value` {only `god` knows it}
 <p align="center">
<img width="800" alt="Screenshot 2023-04-07 at 5 53 25 PM" src="https://user-images.githubusercontent.com/97736991/230608213-56cd90d0-915f-4ba7-8d73-87185509c3d5.png"></p>

### Intuition
* `Bias`: Is the `sampling distribution` centered around the `true` value?
* `Variance`: How scattered is the  `sampling distribution` for different `training sets` 
* The `spread` is a function of size of the `training set`. Larger `training set` leads to lesser `spread` 
* A `high biased` algorithm, no matter how large the `training set` the $\theta$ always keeps awaw from $\theta^{\*}$
* A `high variance` is highly `distracted` by the `noise` in data and easily gets `swayed`. 
#### It is `true` that as $m \to \infty$, $Var\[\hat{\theta}\] \to 0$


### Definitions 
* The rate at which  $Var\[\hat{\theta}\] approaches $0$ as $m$ approached $\infty$ is called `statistical efficiency`
* `If` $\hat{\theta} \to \theta^{\*}$ as $m \to \infty$ `then` the `algorithm` is `Consistent`
* `If` $E\[\hat{\theta}\] = \theta^{\*}$ `for every` $m$ then the `estimator` is called an `unbiased estimator` 

## Resolving `Variance` 
1) Increase `training set size` 
2) Regularisation 

 <p align="center">
<img width="500" alt="Screenshot 2023-04-07 at 6 10 13 PM" src="https://user-images.githubusercontent.com/97736991/230610471-a453291c-c2b1-47f3-aefa-0b57f5d8c43d.png"></p>

* So we might be increasing `bias` in the process of reducing `variance` 



* $g$ is the `best posible hypothesis`
* 

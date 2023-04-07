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


#### We have our `Space` ig `hypothesis` 
 <p align="center">
<img width="400" alt="Screenshot 2023-04-07 at 8 27 53 PM" src="https://user-images.githubusercontent.com/97736991/230630205-caaa90ad-725a-4fa3-9fc8-f86f2d5f9722.png">
</p>

* $H$ is the `class` of `hypothesis`. Like the set of all `Logistic Regression hypothesis` 
* $g$ is the `best posible hypothesis` (the `least` error). It need not lie in $H$
* $h^{\*}$ is the `best` `hypothesis` in $H$
* $\hat{h}$ is any `hypothesis` in $H$ trained from a `fininte` sized `training set` 

#### `Empirical Risk`/ `Empirical Error`
* It's the fraction of `training examples` in which `prediction` is incorrect. 
* Here $n$ is the number of `training examples`
<p align="center">
<img width="200" alt="Screenshot 2023-04-07 at 8 44 33 PM" src="https://user-images.githubusercontent.com/97736991/230632930-3594e2d1-8673-497a-8fed-1d76bdfb0eee.png">
 </p>
 
 ### `Generalisation Error`/ `Risk`
 * It's the `fraction` of `ALL` examples on which `prediction` is incorrect 
 * You can also write in in the `Expectation-indicator notion`
<p align="center">
<img width="200" alt="Screenshot 2023-04-07 at 8 46 24 PM" src="https://user-images.githubusercontent.com/97736991/230633202-f132c0ee-8158-48e8-a6d9-24ff0e8cc2f5.png"> </p>

### `Bayes Error/ Irreducible error` $\epsilon(g)$
* The `error` made by the `best possible` hypothesis. 
* Maybe due to, example $D$ giving `2` different $y^{(i)}$'s for the same $x^{(i)}$. No learning algorithm can fix this.

### `Approximation Error`: $\epsilon(h^{\*}) - \epsilon(g)$
* The `cost` we our paying for limiting ourselves to the `c;ass` we are using.

### `Estimation Error`: $\epsilon(\hat{h}) - \epsilon(h^{\*})$
* The `cost` of training on a `limited data set` 

* `Notice`: $\epsilon(\hat{h})$ = `Bayes Error` + `Approximation Error` + `Estimation Error`
#### `Estimation Error` can be further `broken down` into: `Estimation Bias` + `Estimation Variance`
<p align="center">
<img width="800" alt="Screenshot 2023-04-07 at 8 58 51 PM" src="https://user-images.githubusercontent.com/97736991/230635316-877744c8-08a9-49cf-985d-fa00aed81919.png"></p>

# High Bias:
* Make $H$ bigger so we can `enclose` $g$ in $H$.
* But now the class is `bigger` so the `variance` increases

# High Variance
* Make $H$ smakker but this `might` move $\hat{h}$ away from $g$, hence increasing `bias`




#### What regilarisation does?
* You are `shrinking` the $H$ as we are `penalising` the $\omega$ where  $\lVert \omega \rVert$ is large. 
* Think of it as $\omega = 0$ is present in $H$. 


# `ERM` (`Empirical Risk Mimiser`): Another `Learning Algorithm` 

$$\hat{h}_{ERM} = \underset{h \in H}{argmin} \frac{1}{m} \sum \limits_{1}^{m} 𝟙\\{h(x^{(i)}) \neq y^{(i)}\\}$$

or equivalently

<p align="center">
<img width="325" alt="Screenshot 2023-04-07 at 9 17 27 PM" src="https://user-images.githubusercontent.com/97736991/230637971-202cad87-6fe9-40d6-b372-e510e6ba58e5.png"></p>

* So we we are trying to minimise the `training error`. It is different from the `Maximum Likelihood` we do in `Logistic Regression`
* It can be shown that it can handle losses like `Logistic Losses`




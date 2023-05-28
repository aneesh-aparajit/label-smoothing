# Label Smoothing 

Label Smoothing is a method, which is used do regularization for classification tasks.

## Derivation

Let's first consider the __Cross Entropy Loss Function__:

$$
\mathcal{L} = -\sum_{k=1}^Cq(k|x_i)\log p(k|x_i)
$$

Now, label smoothing is done by adding a weightage to the __Uniform Distribution__.

In the classic Cross Entropy Loss given above, we have the $q(k|x_i) = 1$ if $k$ is the correct class else 0. This might lead to poor generalization on really large datasets. So, what we can do is, try to maintain a distribution, which is uniformly distributed but has more value for the correct class. 

Such a function can be defined as:

$$
q'(k|x_i) = (1 - \epsilon)q(k|x_i) + \frac{\epsilon}{C}
$$

Now, we substitute this to the original function:

$$
\mathcal{L} = -\sum_{k=1}^Cq'(k|x_i)\log p(k|x_i)
$$

$$
\mathcal{L} = -\sum_{k=1}^C[(1-\epsilon)q(k|x_i)+\frac{\epsilon}{C}]\log p(k|x_i)
$$

$$
\mathcal{L} = -\sum_{k=1}^C[(1-\epsilon)q(k|x_i)\log p(k|x_i) + \frac{\epsilon}{C}\log p(k|x_i)]
$$

$$

\mathcal{L} = -\sum_{k=1}^C(1-\epsilon)q(k|x_i)\log p(k|x_i) + \sum_{k=1}^C\frac{\epsilon}{C}\log p(k|x_i)
$$

$$
\mathcal{L} = (1-\epsilon)[-\sum_{k=1}^Cq(k|x_i)\log p(k|x_i)] + \frac{\epsilon}{C}[-\sum_{k=1}^C\log p(k|x_i)]
$$


Now, this is essentially:

```python
loss = (1 - eps)*CrossEntropyLoss(p) + eps/C*(-sum(log_softmax(p)))
```

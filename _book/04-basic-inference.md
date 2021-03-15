# Basic Statistical Inference

 * [One Sample Inference]  
 * [Two Sample Inference]  
 * [Categorical Data Analysis]  

<br>

 * Make **inferences** (an interpretation) about the true parameter value $\beta$ based on our estimator/estimate
 * Test whether our underlying assumptions (about the true population parameters, random variables, or model specification) hold true.


Testing does not  

 * Confirm with 100% a hypothesis is true
 * Confirm with 100% a hypothesis is false
 * Tell you how to interpret the estimate value (Economic vs. Practical vs. Statistical Significance)


Hypothesis: Translate an objective in better understanding the results in terms of specifying a value (or sets of values) in which our population parameters should/should not lie. 

 * **Null hypothesis** ($H_0$): A statement about the population parameter that we take to be true in which we would need the data to provide substantial evidence that against it. 
    + Can be either a single value (ex: $H_0: \beta=0$) or a set of values (ex: $H_0: \beta_1 \ge 0$)
    + Will generally be the value you would not like the population parameter to be (subjective)
        - $H_0: \beta_1=0$ means you would like to see a non-zero coefficient
        - $H_0: \beta_1 \ge 0$ means you would like to see a negative effect
    + "Test of Significance" refers to the two-sided test: $H_0: \beta_j=0$
 * **Alternative hypothesis** ($H_a$ or $H_1$) (Research Hypothesis): All other possible values that the population parameter may be if the null hypothesis does not hold. 


**Type I Error**  

Error made when $H_0$ is rejected when, in fact, $H_0$ is true.  
The probability of committing a Type I error is $\alpha$ (known as **level of significance** of the test)  

Type I error ($\alpha$): probability of rejecting $H_0$ when it is true.  

Legal analogy: In U.S. law, a defendant is presumed to be "innocent until proven guilty".  
If the null hypothesis is that a person is innocent, the Type I error is the probability that you conclude the person is guilty when he is innocent. 


<br>

**Type II Error**  

Type II error level ($\beta$): probability that you fail to reject the null hypothesis when it is false.  

In the legal analogy, this is the probability that you fail to find the person guilty when he or she is guilty. 

Error made when $H_0$ is not rejected when, in fact, $H_1$ is true  
The probability of committing a Type II error is $\beta$ (known as the **power** of the test) 




Random sample of size n: A collection of n independent random variables taken from the distribution X, each with the same distribution as X.  

**Sample mean**   

$$
\bar{X}= (\sum_{i=1}^{n}X_i)/n
$$

**Sample Median**  

$\tilde{x}$ = the middle observation in a sample of observation order from smallest to largest (or vice versa).   

If n is odd, $\tilde{x}$ is the middle observation,  
If n is even, $\tilde{x}$ is the average of the two middle observations.

**Sample variance**
$$
S^2 = \frac{\sum_{i=1}^{n}(X_i = \bar{X})^2}{n-1}= \frac{n\sum_{i=1}^{n}X_i^2 -(\sum_{i=1}^{n}X_i)^2}{n(n-1)}
$$

**Sample standard deviation**
$$
S = \sqrt{S^2}
$$

**Sample proportions**
$$
\hat{p} = \frac{X}{n} = \frac{\text{number in the sample with trait}}{\text{sample size}}
$$


$$
\widehat{p_1-p_2} = \hat{p_1}-\hat{p_2} = \frac{X_1}{n_1} - \frac{X_2}{n_2} = \frac{n_2X_1 = n_1X_2}{n_1n_2}
$$

**Estimators**  
**Point Estimator**  
$\hat{\theta}$ is a statistic used to approximate a population parameter $\theta$

<br>

**Point estimate**  
The numerical value assumed by $\hat{\theta}$ when evaluated for a given sample

<br>

**Unbiased estimator**  
If $E(\hat{\theta}) = \theta$, then $\hat{\theta}$ is an unbiased estimator for $\theta$   

 1. $\bar{X}$ is an unbiased estimator for $\mu$
 2. $S^2$ is an unbiased estimator for $\sigma^2$
 3. $\hat{p}$ is an unbiased estimator for p
 4. $\widehat{p_1-P_2}$ is an unbiased estimator for $p_1- p_2$
 5. $\bar{X_1} - \bar{X_2}$ is an unbiased estimator for $\mu_1 - \mu_2$

**Note**: $S$ is a biased estimator for $\sigma$

**Distribution of the sample mean**  

If $\bar{X}$ is the sample mean based on a random sample of size n drawn from a normal distribution X with mean $\mu$ and standard deviation $\sigma$, the $\bar{X}$ is normally distributed, with mean $\mu_{\bar{X}} = \mu$ and variance $\sigma_{\bar{X}}^2 = Var(\bar{X}) = \frac{\sigma^2}{n}$. Then the **standard error of the mean** is: $\sigma_{\bar{X}}= \frac{\sigma}{\sqrt{n}}$



## One Sample Inference

$Y_i \sim  i.i.d. N(\mu, \sigma^2)$

i.i.d. standards for "independent and identically distributed"

Hence, we have the following model:  

$Y_i=\mu +\epsilon_i$ where  

 * $\epsilon_i \sim^{iid} N(0,\sigma^2)$  
 * $E(Y_i)=\mu$  
 * $Var(Y_i)=\sigma^2$  
 * $\bar{y} \sim N(\mu,\sigma^2/n)$  

<br>

### The Mean

When $\sigma^2$ is estimated by $s^2$, then  

$$
\frac{\bar{y}-\mu}{s/\sqrt{n}} \sim t_{n-1}
$$

Then, a $100(1-\alpha) \%$ confidence interval for $\mu$ is obtained from:  

$$
1 - \alpha = P(-t_{\alpha/2;n-1} \le \frac{\bar{y}-\mu}{s/\sqrt{n}} \le t_{\alpha/2;n-1}) \\
= P(\bar{y} - (t_{\alpha/2;n-1})s/\sqrt{n} \le \mu \le \bar{y} + (t_{\alpha/2;n-1})s/\sqrt{n})
$$

And the interval is  

$$
\bar{y} \pm (t_{\alpha/2;n-1})s/\sqrt{n}
$$

and $s/\sqrt{n}$ is the standard error of $\bar{y}$  

If the experiment were repeated many times, $100(1-\alpha) \%$ of these intervals would contain $\mu$


| | Confidence Interval $100(1-\alpha)%$ | Sample Sizes <br> Confidence $\alpha$, Error d | Hypothesis Testing <br> Test Statistic |
|---|---|---|---|
When $\sigma^2$ is known, X is normal (or $n \ge 25$) | $\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$ | $n \approx \frac{z_{\alpha/2}^2 \sigma^2}{d^2}$| $z = \frac{\bar{X}-\mu_0}{\sigma/\sqrt{n}}$ |
When $\sigma^2$ is unknown, X is normal (or $n \ge 25$) | $\bar{X} \pm t_{\alpha/2}\frac{s}{\sqrt{n}}$ | $n \approx \frac{z_{\alpha/2}^2 s^2}{d^2}$| $t = \frac{\bar{X}-\mu_0}{s/\sqrt{n}}$ |


#### For Difference of Means ($\mu_1-\mu_2$), Independent Samples

| | $100(1-\alpha)%$ Confidence Interval | Hypothesis Testing <br> Test Statistic | |
|---|---|---|---|
|When $\sigma^2$ is known | $\bar{X}_1 - \bar{X}_2 \pm z_{\alpha/2}\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}$ | $z= \frac{(\bar{X}_1-\bar{X}_2)-(\mu_1-\mu_2)_0}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}$| |
|When $\sigma^2$ is unknown, Variances Assumed EQUAL |  $\bar{X}_1 - \bar{X}_2 \pm t_{\alpha/2}\sqrt{s^2_p(\frac{1}{n_1}+\frac{1}{n_2})}$ | $t = \frac{(\bar{X}_1-\bar{X}_2)-(\mu_1-\mu_2)_0}{\sqrt{s^2_p(\frac{1}{n_1}+\frac{1}{n_2})}}$| Pooled Variance: $s_p^2 = \frac{(n_1 -1)s^2_1 - (n_2-1)s^2_2}{n_1 + n_2 -2}$ <br> Degrees of Freedom: $\gamma = n_1 + n_2 -2$|
|When $\sigma^2$ is unknown, Variances Assumed UNEQUAL |  $\bar{X}_1 - \bar{X}_2 \pm t_{\alpha/2}\sqrt{(\frac{s^2_1}{n_1}+\frac{s^2_2}{n_2})}$ | $t = \frac{(\bar{X}_1-\bar{X}_2)-(\mu_1-\mu_2)_0}{\sqrt{(\frac{s^2_1}{n_1}+\frac{s^2_2}{n_2})}}$| Degrees of Freedom: $\gamma = \frac{(\frac{s_1^2}{n_1}+\frac{s^2_2}{n_2})^2}{\frac{(\frac{s_1^2}{n_1})^2}{n_1-1}+\frac{(\frac{s_2^2}{n_2})^2}{n_2-1}}$|


#### For Difference of Means ($\mu_1 - \mu_2$), Paired Samples (D = X-Y)  

**$100(1-\alpha)%$ Confidence Interval**  
$$
\bar{D} \pm t_{\alpha/2}\frac{s_d}{\sqrt{n}}
$$

**Hypothesis Testing Test Statistic**  

$$
t = \frac{\bar{D}-D_0}{s_d / \sqrt{n}}
$$



#### Difference of Two Proprotions


**Mean** 

$$
\hat{p_1}-\hat{p_2}
$$

**Variance**
$$
\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}
$$

**$100(1-\alpha)%$ Confidence Interval**  

$$
\hat{p_1}-\hat{p_2} + z_{\alpha/2}\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}
$$

**Sample Sizes, Confidence $\alpha$, Error d**  
(Prior Estimate fo $\hat{p_1},\hat{p_2}$)

$$
n \approx \frac{z_{\alpha/2}^2[p_1(1-p_1)+p_2(1-p_2)]}{d^2}
$$

(No Prior Estimates for $\hat{p}$)

$$
n \approx \frac{z_{\alpha/2}^2}{2d^2}
$$

**Hypothesis Testing - Test Statistics**  

Null Value $(p_1 - p_2) \neq 0$

$$
z = \frac{(\hat{p_1} - \hat{p_2})-(p_1 - p_2)_0}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}}
$$


Null Value $(p_1 - p_2)_0 = 0$

$$
z = \frac{\hat{p_1} - \hat{p_2}}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1}+\frac{1}{n_2})}}
$$

where   

$$
\hat{p}= \frac{x_1 + x_2}{n_1 + n_2} = \frac{n_1 \hat{p_1} + n_2 \hat{p_2}}{n_1 + n_2}
$$




### Single Variance

$$
1 - \alpha = P( \chi_{1-\alpha/2;n-1}^2) \le (n-1)s^2/\sigma^2 \le \chi_{\alpha/2;n-1}^2) \\
= P(\frac{(n-1)s^2}{\chi_{\alpha/2}^2} \le \sigma^2 \le \frac{(n-1)s^2}{\chi_{1-\alpha/2}^2})
$$

and a $100(1-\alpha) \%$ confidence interval for $\sigma^2$ is:  

$$
(\frac{(n-1)s^2}{\chi_{\alpha/2;n-1}^2},\frac{(n-1)s^2}{\chi_{1-\alpha/2;n-1}^2})
$$
Confidence limits for $\sigma^2$ are obtained by computing the positive square roots of these limits

Equivalently,  

**$100(1-\alpha)%$ Confidence Interval**  

$$
L_1 = \frac{(n-1)s^2}{\chi^2_{\alpha/2}} \\
L_1 = \frac{(n-1)s^2}{\chi^2_{1-\alpha/2}}
$$
**Hypothesis Testing Test Statistic**

$$
\chi^2 = \frac{(n-1)s^2}{\sigma^2_0}
$$


### Single Proportion (p)

| Confidence Interval $100(1-\alpha)%$ | Sample Sizes <br> Confidence $\alpha$, Error d (prior estimate for $\hat{p}$) | (No prior estimate for $\hat{p}$)|Hypothesis Testing <br> Test Statistic |
|---|---|---|---|
|$\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ | $n \approx \frac{z_{\alpha/2}^2 \hat{p}(1-\hat{p})}{d^2}$| $n \approx \frac{z_{\alpha/2}^2}{4d^2}$ |$z = \frac{\hat{p}-p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$ |


### Power 

Formally, power (for the test of the mean) is given by:  

$$
\pi(\mu) = 1 - \beta = P(\text{test rejects } H_0|\mu)
$$
To evaluate the power, one needs to know the distribution of the test statistic if the null hypothesis is false.  

For 1-sided z-test where $H_0: \mu \le \mu_0 \\ H_A: \mu >0$  

The power is:  

$$
\begin{align}
\pi(\mu) &= P(\bar{y} > \mu_0 + z_{\alpha} \sigma/\sqrt{n}|\mu) \\
&= P(Z = \frac{\bar{y} - \mu}{\sigma / \sqrt{n}} > z_{\alpha} + \frac{\mu_0 - \mu}{\sigma/ \sqrt{n}}|\mu) \\
&= 1 - \Phi(z_{\alpha} + \frac{(\mu_0 - \mu)\sqrt{n}}{\sigma}) \\
&= \Phi(-z_{\alpha}+\frac{(\mu -\mu_0)\sqrt{n}}{\sigma})
\end{align}
$$

where $1-\Phi(x) = \Phi(-x)$ since the normal pdf is symmetric  

Power is correlated to the difference in $\mu - \mu_0$, sample size n, variance $\sigma^2$, and the $\alpha$-level of the test (through $z_{\alpha}$)  
Equivalently, power can be increased by making $\alpha$ large, $\sigma^2$ smaller, or n larger. 

For 2-sided z-test is:  

$$
\pi(\mu) = \Phi(-z_{\alpha/2} + \frac{(\mu_0 - \mu)\sqrt{n}}{\sigma}) + \Phi(-z_{\alpha/2}+\frac{(\mu - \mu_0)\sqrt{n}}{\sigma})
$$



### Sample Size

#### 1-sided Z-test

Example: to show that the mean response $\mu$ under the treatment is higher than the mean response $\mu_0$ without treatment (show that the treatment effect $\delta = \mu -\mu_0$ is large)

Because power is an increasing function of $\mu - \mu_0$, it is only necessary to find n that makes the power equal to $1- \beta$ at $\mu = \mu_0 + \delta$  

Hence, we have 

$$
\pi(\mu_0 + \delta) = \Phi(-z_{\alpha} + \frac{\delta \sqrt{n}}{\sigma}) = 1 - \beta
$$

Since $\Phi (z_{\beta})= 1-\beta$, we have  

$$
-z_{\alpha} + \frac{\delta \sqrt{n}}{\sigma} = z_{\beta}
$$

Then n is 

$$
n = (\frac{(z_{\alpha}+z_{\beta})\sigma}{\delta})^2
$$

Then, we need larger samples, when  

 * the sample variability is large ($\sigma$ is large)
 * $\alpha$ is small ($z_{\alpha}$ is large)
 * power $1-\beta$ is large ($z_{\beta}$ is large)
 * The magnitude of the effect is smaller ($\delta$ is small) 

Since we don't know $\delta$ and $\sigma$. We can base $\sigma$ on previous studies, pilot studies. Or, obtain an estimate of $\sigma$ by anticipating the range of the observation (without outliers). divide this range by 4 and use the resulting number as an approximate estimate of $\sigma$. For normal (distribution) data, this is reasonable. 


#### 2-sided Z-test

We want to know the min n, required to guarantee $1-\beta$ power when the treatment effect $\delta = |\mu - \mu_0|$ is at least greater than 0. Since the power function for the 2-sided is increasing and symmetric in $|\mu - \mu_0|$, we only need to find n that makes the power equal to $1-\beta$ when $\mu = \mu_0 + \delta$  

$$
n = (\frac{(z_{\alpha/2} + z_{\beta}) \sigma}{\delta})^2
$$

We could also use the confidence interval apporach. If we reuqire that an $\alpha$-level two-cided CI for $\mu$ be 

$$
\bar{y} \pm D
$$
where $D = z_{\alpha/2}\sigma/\sqrt{n}$ gives  

$$
n = (\frac{z_{\alpha/2}\sigma}{D})^2
$$
(round up to the nearest integer)  



```r
data = rnorm(100)
t.test(data, conf.level=0.95)
```

```
## 
## 	One Sample t-test
## 
## data:  data
## t = -1.7389, df = 99, p-value = 0.08517
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.3848177  0.0253590
## sample estimates:
##  mean of x 
## -0.1797294
```

$$
H_0: \mu \ge 30 \\
H_a: \mu < 30
$$


```r
t.test(data, mu=30,alternative="less")
```

```
## 
## 	One Sample t-test
## 
## data:  data
## t = -291.99, df = 99, p-value < 2.2e-16
## alternative hypothesis: true mean is less than 30
## 95 percent confidence interval:
##          -Inf -0.008111588
## sample estimates:
##  mean of x 
## -0.1797294
```






### Note

For t-tests, the sample and power are not as easy as z-test. 

$$
\pi(\mu) = P(\frac{\bar{y}-\mu_0}{s/\sqrt{n}}> t_{n-1;\alpha}|\mu)
$$

when $\mu > \mu_0$ (i.e., $\mu - \mu_0 = \delta$), the random variable $(\bar{y} - \mu_0)/(s/\sqrt{n})$ does not have a [Student's t distribution][Student T], but rather is distributed as a non-central t-distribution with non-centrality parameter $\delta \sqrt{n}/\sigma$ and d.f. of $n-1$  

 * The power is an increasing function of this non-centrality parameter (note, when $\delta = 0$ the distribution is usual Student's t-distribution). 
 * To evaluate power, one must consider numerical procedure or use special charts 

Approximate Sample Size Adjustment for t-test. We use an adjustment to the z-test determination for sample size.  

Let $v = n-1$, where n is sample size derived based on the z-test power. Then the 2-sided t-test sample size (apporximate) is given:  

$$
n^* = \frac{(t_{v;\alpha/2}+t_{v;\beta})^2 \sigma^2}{\delta^2}
$$

### One-sample Non-parametric Methods


```r
lecture.data=c(0.76, 0.82, 0.80, 0.79, 1.06, 0.83, -0.43, -0.34, 3.34, 2.33)
```


#### Sign Test 

If we want to test $H_0: \mu_{(0.5)} = 0; H_a: \mu_{(0.5)} >0$ where $\mu_{(0.5)}$ is the population median. We can  

 (1) Count the number of observation ($y_i$'s) that exceed 0. Denote this number by $s_+$, called the number of plus signs. Let $s_- = n - s_+$, which is the number of minus signs. 
 (2) Reject $H_0$ if $s_+$ is large or equivalently, if $s_-$ is small.  

To determine how large $s_+$ must be to reject $H_0$ at a given significance level, we need to know the distribution of the corresponding random variable $S_+$ under the null hypothesis, which is a [binomial][Binomial] with p = 1/2,w hen the null is true.  

To work out the null distribution using the binomial formula, we have $\alpha$-level test rejects $H_0$ if $s_+ \ge b_{n,\alpha}$, where $b_{n,\alpha}$ is the upper $\alpha$ critical point of the $Bin(n,1/2)$ distribution. Both $S_+$ and $S_-$ have this same distribution ($S = S_+ = S_-$).  

$$
\text{p-value} = P(S \ge s_+) = \sum_{i = s_+}^{n} {{n}\choose{i}} (\frac{1}{2})^n
$$
equivalently, 

$$
P(S \le s_-) = \sum_{i=0}^{s_-}{{n}\choose{i}} (\frac{1}{2})^2
$$
For large sample sizes, we could use the normal approximation for the binomial, in which case reject $H_0$ if 

$$
s_+ \ge n/2 + 1/2 + z_{\alpha}\sqrt{n/4}
$$

For the 2-sided test, we use the tests statistic $s_{max} = max(s_+,s_-)$ or $s_{min} = min(s_+, s_-)$. An $\alpha$-level test rejects $H_0$ if the p-value is $\le \alpha$, where the p-value is computed from:  

$$
p-value = 2 \sum_{i=s_{max}}^{n} {{n}\choose{i}} (\frac{1}{2})^n = s \sum_{i=0}^{s_{min}} {{n}\choose{i}} (\frac{1}{2})^n
$$
Equivalently, rejecting $H_0$ if $s_{max} \ge b_{n,\alpha/2}$ 

A large sample normal approximation can be used, where  

$$
z = \frac{s_{max}- n/2 -1/2}{\sqrt{n/4}}
$$
and reject $H_0$ at $\alpha$ if $z \ge z_{\alpha/2}$  

However, treatment of 0 is problematic for this test.  

 * Solution 1: randomly assign 0 to the positive or negative (2 researchers might get different results).  
 * Solution 2: count each 0 as a contribution 1/2 toward $s_+$ and $s_-$ (but then could not apply the [binomial][Binomial] distribution)  
 * Solution 3: ignore 0 (reduces the power of test due to decreased sample size). 



```r
binom.test(sum(lecture.data > 0), length(lecture.data)) # alternative = "greater" or alternative = "less"
```

```
## 
## 	Exact binomial test
## 
## data:  sum(lecture.data > 0) and length(lecture.data)
## number of successes = 8, number of trials = 10, p-value = 0.1094
## alternative hypothesis: true probability of success is not equal to 0.5
## 95 percent confidence interval:
##  0.4439045 0.9747893
## sample estimates:
## probability of success 
##                    0.8
```



#### Wilcoxon Signed Rank Test

Since the [Sign Test] could not consider the magnitude of each observation from 0, the [Wilcoxon Signed Rank Test] improves by  taking account the ordered magnitudes of the observation, but it will impose the requirement of symmetric to this test (while [Sign Test] does not)

$$
H_0: \mu_{0.5} = 0 \\
H_a: \mu_{0.5} > 0
$$
(assume no ties or same observations)  

The signed rank test procedure:  

 1. rank order the observation $y_i$ in terms of their absolute values. Let $r_i$ be the rank of $y_i$ in this ordering. Since we assume no ties, the ranks $r_i$ are uniquely determined and are a permutation of the integers 1,2,...,n.  
 2. Calculate $w_+$, which is the sum of the ranks of the positive values, and $w_-$, which is the sum of the ranks of the negative values. Note that $w_+ + w_- = r_1 + r_2 + ... = 1 + 2 + ... + n = n(n+1)/2$  
 3. Reject $H_0$ if $w_+$ is large (or if $w_-$ is small)  
 
To know what is large or small with regard to $w_+$ and $w_-$, we need the distribution of $W_+$ and $W_-$ when the null is true.  

Since these null distributions are identical and symmetric, the p-value is $P(W \ge w_+) = P(W \le w_-)$  

An $\alpha$-level test rejects the null if the p-value is $\le \alpha$, or if $w_+ \ge w_{n,\alpha}$, where $w_{n,\alpha}$ is the upper $\alpha$ critical point of the null distribution of W.  

This distribution of W has a special table. For large n, the distribution of W is approximately normal.  

$$
z = \frac{w_+ - n(n+1) /4 -1/2}{\sqrt{n(n+1)(2n+1)/24}}
$$

The test rejcets $H_0$ at level $\alpha$ if  

$$
w_+ \ge n(n+1)/4 +1/2 + z_{\alpha}\sqrt{n(n+1)(2n+1)/24} \approx w_{n,\alpha}
$$

For the 2-sided test, we use $w_{max}=max(w_+,w_-)$ or $w_{min}=min(w_+,w_-)$, with p-value given by: 

$$
p-value = 2P(W \ge w_{max}) = 2P(W \le w_{min})
$$
Same as [Sign Test],we ignore 0. In some cases where some of the $|y_i|$'s may be tied for the same rank, we simply assign each of the tied ranks the average rank (or "midrank").  

Example, if $y_1 = -1$, $y_3 = 3$ and $y_3 = -3$, and $y_4 =5$, then $r_1 = 1$, $r_2 = r_3=(2+3)/2 = 2.5$, $r_4 = 4$


```r
wilcox.test(lecture.data) #does not use normal approximation (using the underlying W distribution)
```

```
## 
## 	Wilcoxon signed rank exact test
## 
## data:  lecture.data
## V = 52, p-value = 0.009766
## alternative hypothesis: true location is not equal to 0
```

```r
wilcox.test(lecture.data,exact=F) #uses normal approximation
```

```
## 
## 	Wilcoxon signed rank test with continuity correction
## 
## data:  lecture.data
## V = 52, p-value = 0.01443
## alternative hypothesis: true location is not equal to 0
```



## Two Sample Inference
### Means

Suppose we have 2 sets of observations,  

 * $y_1,..., y_{n_y}$  
 * $x_1,...,x_{n_x}$

that are random samples from two independent populations with means $\mu_y$ and $\mu_x$ and variances $\sigma^2_y$,$\sigma^2_x$. 
Our goal is to compare $\mu_x$ and $\mu_y$ or $\sigma^2_y = \sigma^2_x$



#### Large Sample Tests

Assume that $n_y$ and $n_x$ are large ($\ge 30$). Then,  

$$
E(\bar{y} - \bar{x}) = \mu_y - \mu_x \\
Var(\bar{y} - \bar{x}) = \sigma^2_y /n_y + \sigma^2_x/n_x
$$

Then,  

$$
Z = \frac{\bar{y}-\bar{x} - (\mu_y - \mu_x)}{\sqrt{\sigma^2_y /n_y + \sigma^2_x/n_x}} \sim N(0,1)
$$
(according to [Central Limit Theorem]). For large samples, we can replace variances by their unbiased estimators ($s^2_y,s^2_x$), and get the same large sample distribution.   

An approximate $100(1-\alpha) \%$ CI for $\mu_y - \mu_x$ is given by:  

$$
\bar{y} - \bar{x} \pm z_{\alpha/2}\sqrt{s^2_y/n_y + s^2_x/n_x}
$$

$$
H_0: \mu_y - \mu_x = \delta_0 \\
H_A: \mu_y - \mu_x \neq \delta_0
$$

at the $\alpha$-level with the statistic:  

$$
z = \frac{\bar{y}-\bar{x} - \delta_0}{\sqrt{s^2_y /n_y + s^2_x/n_x}}
$$

and reject $H_0$ if $|z| > z_{\alpha/2}$  

If $\delta = )$, it means that we are testing whether two means are equal.


#### Small Sample Tests

If the two samples are from normal distribution, iid $N(\mu_y,\sigma^2_y)$ and iid $N(\mu_x,\sigma^2_x)$ and the two samples are independent, we can do inference based on the [t-distribution][Student T]  

Then we have 2 cases  

 * [Equal Variance]
 * [Unequal Variance]

##### Equal variance

**Assumptions**  

 * iid: so that $var(\bar{y}) = \sigma^2_y / n_y ; var(\bar{x}) = \sigma^2_x / n_x$  
 * Independence between samples: No observation from one sample can influence any observation from the other sample, to have  

$$
\begin{align}
var(\bar{y} - \bar{x}) &= var(\bar{y}) + var{\bar{x}} - 2cov(\bar{y},\bar{x}) \\
&= var(\bar{y}) + var{\bar{x}} \\
&= \sigma^2_y / n_y + \sigma^2_x / n_x 
\end{align}
$$

 * Normality: Justifies the use of the [t-distribution][Student T]

Let $\sigma^2 = \sigma^2_y = \sigma^2_x$. Then, $s^2_y$ and $s^2_x$ are both unbiased estimators of $\sigma^2$. We then can pool them.  

Then the pooled variance estimate is 
$$
s^2 = \frac{(n_y - 1)s^2_y + (n_x - 1)s^2_x}{(n_y-1)+(n_x-1)}
$$
has $n_y + n_x -2$ df.  

Then the test statistic  

$$
T = \frac{\bar{y}- \bar{x} -(\mu_y - \mu_x)}{s\sqrt{1/n_y + 1/n_x}} \sim t_{n_y + n_x -2}
$$

$100(1 - \alpha) \%$ CI for $\mu_y - \mu_x$ is

$$
\bar{y} - \bar{x} \pm (t_{n_y + n_x -2})s\sqrt{1/n_y + 1/n_x}
$$

Hypothesis testing:  
$$
H_0: \mu_y - \mu_x = \delta_0 \\
H_1: \mu_y - \mu_x \neq \delta_0
$$

we reject $H_0$ if $|t| > t_{n_y + n_x -2;\alpha/2}$


##### Unequal Variance

**Assumptions**  

 1. Two samples are independent  
        1. Scatter plots  
        2. [Correlation coefficient (if normal)][Correlation Coefficient with Normal Probability Plots]
 2. Independence of observation in each sample  
        1. Test for serial correlation  
 3. For each sample, homogeneity of variance  
        1. Scatter plots  
        2. Formal tests  
 4. [Normality][Normality Assessment]  
 5. Equality of variances (homogeneity of variance between samples)  
        1. [F-test]  
        2. Barlett test  
        3. [Modified Levene Test]  


To compare 2 normal $\sigma^2_y \neq \sigma^2_x$, we use the test statistic:  

$$
T = \frac{\bar{y}- \bar{x} -(\mu_y - \mu_x)}{\sqrt{s^2_y/n_y + s^2_x/n_x}} 
$$
In this case, T does not follow the [t-distribution][Student T] (its distribution depends on the ratio of the unknown variances $\sigma^2_y,\sigma^2_x$). In the case of small sizes, we can can approximate tests by using the Welch-Satterthwaite method [@Satterthwaite_1946]. We assume T can be approximated by a [t-distribution][Student T], and adjust the degrees of freedom.  

Let $w_y = s^2_y /n_y$ and $w_x = s^2_x /n_x$ (the w's are the square of the respective standard errors)  
Then, the degrees of freedom are  

$$
v = \frac{(w_y + w_x)^2}{w^2_y / (n_y-1) + w^2_x / (n_x-1)}
$$

Since v is usually fractional, we truncate down to the nearest integer.  

$100 (1-\alpha) \%$ CI for $\mu_y - \mu_x$ is 

$$
\bar{y} - \bar{x} \pm t_{v,\alpha/2} \sqrt{s^2_y/n_y + s^2_x /n_x}
$$

Reject $H_0$ if $|t| > t_{v,\alpha/2}$, where 

$$
t = \frac{\bar{y} - \bar{x}-\delta_0}{\sqrt{s^2_y/n_y + s^2_x /n_x}}
$$


### Variances


$$
F_{ndf,ddf}= \frac{s^2_1}{s^2_2}
$$

where $s^2_1>s^2_2, ndf = n_1-1,ddf = n_2-1$



#### F-test

Test 

$$
H_0: \sigma^2_y = \sigma^2_x \\
H_a: \sigma^2_y \neq \sigma^2_x
$$

Consider the test statistic,  

$$
F= \frac{s^2_y}{s^2_x}
$$

Reject $H_0$ if  

 * $F>f_{n_y -1,n_x -1,\alpha/2}$ or  
 * $F<f_{n_y -1,n_x -1,1-\alpha/2}$  

Where $F>f_{n_y -1,n_x -1,\alpha/2}$ and $F<f_{n_y -1,n_x -1,1-\alpha/2}$ are the upper and lower $\alpha/2$ critical points of an [F-distribution][F-Distribution], with a $n_y-1$ and $n_x-1$ degrees of freedom.  

**Note**  

 * This test depends heavily on the assumption Normality.  
 * In particular, it could give to many significant results when observations come from long-tailed distributions (i.e., positive kurtosis).  
 * If we cannot find support for [normality][Normality Assessment], then we can use nonparametric tests such as the [Modified Levene Test] 
 


```r
data(iris)
irisVe=iris$Petal.Width[iris$Species=="versicolor"] 
irisVi=iris$Petal.Width[iris$Species=="virginica"]

var.test(irisVe,irisVi)
```

```
## 
## 	F test to compare two variances
## 
## data:  irisVe and irisVi
## F = 0.51842, num df = 49, denom df = 49, p-value = 0.02335
## alternative hypothesis: true ratio of variances is not equal to 1
## 95 percent confidence interval:
##  0.2941935 0.9135614
## sample estimates:
## ratio of variances 
##          0.5184243
```



#### Modified Levene Test (Brown-Forsythe Test)

 * considers averages of absolute deviations rather than squared deviations. Hence, less sensitive to long-tailed distributions.  
 * This test is still good for normal data 

For each sample, we consider the absolute deviation of each observation form the median:  

$$
d_{y,i} = |y_i - y_{.5}| \\
d_{x,i} = |x_i - x_{.5}|
$$
Then,  

$$
t_L^* = \frac{\bar{d}_y-\bar{d}_x}{s \sqrt{1/n_y + 1/n_x}}
$$

The pooled variance $s^2$ is given by: 

$$
s^2 = \frac{\sum_i^{n_y}(d_{y,i}-\bar{d}_y)^2 + \sum_j^{n_x}(d_{x,i}-\bar{d}_x)^2}{n_y + n_x -2}
$$

 * If the error terms have constant variance and $n_y$ and $n_x$ are not extremely small, then $t_L^* \sim t_{n_x + n_y -2}$  
 * We reject the null hypothesis when $|t_L^*| > t_{n_y + n_x -2;\alpha/2}$  
 * This is just the two-sample t-test applied to the absolute deviations.   


```r
dVe=abs(irisVe-median(irisVe)) 
dVi=abs(irisVi-median(irisVi)) 
t.test(dVe,dVi,var.equal=T)
```

```
## 
## 	Two Sample t-test
## 
## data:  dVe and dVi
## t = -2.5584, df = 98, p-value = 0.01205
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -0.12784786 -0.01615214
## sample estimates:
## mean of x mean of y 
##     0.154     0.226
```

```r
# small samples t-test  
t.test(irisVe,irisVi,var.equal=F)
```

```
## 
## 	Welch Two Sample t-test
## 
## data:  irisVe and irisVi
## t = -14.625, df = 89.043, p-value < 2.2e-16
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -0.7951002 -0.6048998
## sample estimates:
## mean of x mean of y 
##     1.326     2.026
```


### Power

Consider $\sigma^2_y = \sigma^2_x = \sigma^2$  
Under the assumption of equal variances, we take size samples from both groups ($n_y = n_x = n$)  

For 1-sided testing,  

$$
H_0: \mu_y - \mu_x \le 0 \\
H_a: \mu_y - \mu_x > 0
$$

$\alpha$-level z-test rejects $H_0$ if  

$$
z = \frac{\bar{y} - \bar{x}}{\sigma \sqrt{2/n}} > z_{\alpha}
$$

$$
\pi(\mu_y - \mu_x) = \Phi(-z_{\alpha} + \frac{\mu_y -\mu_x}{\sigma}\sqrt{n/2})
$$

We need sample size n that gie at least $1-\beta$ power when $\mu_y - \mu_x = \delta$, where $\delta$ is the smallest difference that we want to see.  

Power is given by:  

$$
\Phi(-z_{\alpha} + \frac{\delta}{\sigma}\sqrt{n/2}) = 1 - \beta
$$

### Sample Size

Then, the sample size is  

$$
n = 2(\frac{\sigma (z_{\alpha} + z_{\beta}}{\delta})^2
$$

For 2-sided test, replace $z_{\alpha}$ with $z_{\alpha/2}$.  
As with the one-sample case, to perform an exact 2-sample t-test sample size calculation, we must use a non-central [t-distribution][Student T].  

A correction that gives the approximate t-test sample size can be obtained by using the z-test n value in the formula:  
$$
n^* = 2(\frac{\sigma (t_{2n-2;\alpha} + t_{2n-2;\beta})}{\delta})^2
$$

where we use $alpha/2$ for the two-sided test  

### Matched Pair Designs

We have two treatments  

Subject | Treatment A | Treatment B | Difference 
---|---|---|---
1 | $y_1$ | $x_1$ | $d_1 = y_1 - x_1$
2 | $y_2$ | $x_2$ | $d_2 = y_2 - x_2$
. |. |.|.
n | $y_n$ | $x_n$ | $d_n = y_n - x_n$

we assume $y_i \sim^{iid} N(\mu_y, \sigma^2_y)$ and $x_i \sim^{iid} N(\mu_x,\sigma^2_x)$, but since $y_i$ and $x_i$ are measured on the same subject, they are correlated.  

Let  

$$
\mu_D = E(y_i - x_i) = \mu_y -\mu_x \\
\sigma^2_D = var(y_i - x_i) = Var(y_i) + Var(x_i) -2cov(y_i,x_i)
$$

If the matching induces **positive** correlation, then the variance of the difference of the measurements is reduced as compared to the independent case. This is the point of [Matched Pair Designs]. Although covariance can be negative, giving a larger variance of the difference than the independent sample case, usually the covariance is positive. This means both $y_i$ and $x_i$  are large for many of the same subjects, and for others, both measurement are small. (we still assume that various subjects respond independently of each other, which is necessary for the iid assumption within groups).  

Let $d_i = y_i - x_i$, then   

 * $\bar{d} = \bar{y}-\bar{x}$ is the sample mean of the $d_i$  
 * $s_d^2=\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2$ is the sample variance of the difference  

Once the data are converted to differences,  we are back to [One Sample Inference] and can use its tests and CIs. 

### Nonparametric Tests for Two Samples

For [Matched Pair Designs], we can use the [One-sample Non-parametric Methods].  

Assume that Y and X are random variables with CDF $F_y$ and $F_x$. then, Y is **stochastically** larger than X for all real number u, $P(Y > u) \ge P(X > u)$.  

Equivalently, $P(Y \le u) \le P(X \le u)$, which is $F_Y(u) \le F_X(u)$, same thing as $F_Y < F_X$  

If two distributions are identical, except that one is shifted relative to the other, then each of distribution can be indexed by a location parameter, say $\theta_y$ and $\theta_x$. In this case, $Y>X$ if $\theta_y > \theta_x$  

Consider the hypotheses,  

$$
H_0: F_Y = F_X \\
H_a: F_Y < F_X
$$
where the alternative is an upper one-sided alternative.  

 * We can also consider the lower one-sided alternative  

$$
H_a: F_Y > F_X \text{ or} \\
H_a: F_Y < F_X \text{ or } F_Y > F_X
$$

 * In this case, we don't use $H_a: F_Y \neq F_X$ as that allows arbitrary differences between the distributions, without requiring one be stochastically larger than the other.  

If the distributions only differ in terms of their location parameters, we can focus hypothesis tests on the parameters (e.g., $H_0: \theta_y = \theta_x$ vs. $\theta_y > \theta_x$)  

We have 2 equivalent nonparametric tests that consider the hypothesis mentioned above  

 1. [Wilcoxon rank test]  
 2. [Mann-Whitney U test]

#### Wilcoxon rank test

 1. Combine all $n= n_y + n_x$ observations and rank them in ascending order.  
 2. Sum the ranks of the y's and x's separately. Let $w_y$ and $w_x$ be these sums. ($w_y + w_x = 1 + 2 + ... + n = n(n+1)/2$)  
 3. Reject $H_0$ if $w_y$ is large (equivalently, $w_x$ is small)  


Under $H_0$, any arrangement of the y's and x's is equally likely to occur, and there are $(n_y + n_x)!/(n_y! n_x!)$ possible arrangements.  

 * Technically, for each arrangement we can compute the values of $w_y$ and $w_x$, and thus generate the distribution of the statistic under the null hypothesis.  
 * This could lead to computationally intensive.  




```r
wilcox.test(irisVe,irisVi,alternative="two.sided",conf.level=0.95, exact=F,correct=T)
```

```
## 
## 	Wilcoxon rank sum test with continuity correction
## 
## data:  irisVe and irisVi
## W = 49, p-value < 2.2e-16
## alternative hypothesis: true location shift is not equal to 0
```




#### Mann-Whitney U test


The Mann-Whitney test is computed as follows:  

 1. Compare each $y_i$ wiht each $x_i$.  
        Let $u_y$ be the number of pairs in which $y_i > x_i$ 
        Let $u_x$ be the number of pairs in which $y_i < x_i$. (assume there are no ties). 
        There are $n_y n_x$ such comparisons and $u_y + u_x = n_y n_x$.  
 2. Reject $H_0$ if $u_y$ is large (or $u_x$ is small)  


[Mann-Whitney U test] and [Wilcoxon rank test] are related:  
$$
u_y = w_y - n_y(n_y+1) /2 \\
u_x = w_x - n_x(n_x +1)/2
$$

An $\alpha$-level test rejects $H_0$ if $u_y \ge u_{n_y,n_x,\alpha}$, where $u_{n_y,n_x,\alpha}$ is the upper $\alpha$ critical point of the null distribution of the random variable, U.  

The p-value is defined to be $P(Y \ge u_y) = P(U \le u_x)$. One advantage of [Mann-Whitney U test] is that we can use either $u_y$ or $u_x$ to carry out the test.  

For large $n_y$ and $n_x$, the null distribution of U can be well approximated by a normal distribution with mean $E(U) = n_y n_x /2$ and variance $var(U) = n_y n_x (n+1)/12$. A large sample z-test can be based on the statistic:  

$$
z = \frac{u_y - n_y n_x /2 -1/2}{\sqrt{n_y n_x (n+1)/12}}
$$

The test rejects $H_0$ at level $\alpha$ if $z \ge z_{\alpha}$ or if $u_y \ge u_{n_y,n_x,\alpha}$ where  

$$
u_{n_y, n_x, \alpha} \approx n_y n_x /2 + 1/2 + z_{\alpha}\sqrt{n_y n_x (n+1)/12}
$$

For the 2-sided test, we use the test statistic $u_{max} = max(u_y,u_x)$ and $u_{min} = min(u_y, u_x)$ and p-value is given by  

$$
p-value = 2P(U \ge u_{max}) = 2P(U \le u_{min})
$$
Since we assume there are no ties (when $y_i = x_j$), we count 1/2 towards both $u_y$ and $u_x$. Even though the sampling distribution is not the same, but large sample approximation is still reasonable,  


## Categorical Data Analysis

[Categorical Data Analysis] when we have categorical outcomes  

 * Nominal variables: no logical ordering (e.g., sex)  
 * Ordinal variables: logical order, but relative distances between values are not clear (e.g., small, medium, large)  

The distribution of one variable changes when the level (or values) of the other variable change. The row percentages are different in each column. 

### Inferences for Small Samples

The approximate tests based on the asymptotic normality of $\hat{p}_1 - \hat{p}_2$ do not apply for small samples.  

Using **Fisher's Exact Test** to evaluate $H_0: p_1 = p_2$  

 * Assume $X_1$ and $X_2$ are independent [Binomial]  
 * Let $x_1$ and $x_2$ be the corresponding observed values. 
 * Let $n= n_1 + n_2$ be the total sample size
 * $m = x_1 + x_2$ be the observed number of successes.  
 * By assuming that m (total successes) is fixed, and conditioning on this value, one can show that the conditional distribution of the number of successes from sample 1 is [Hypergeometric]  
 * If we want to test $H_0: p_1 = p_2$ and $H_a: p_1 \neq p_2$, we have  

$$
Z^2 = (\frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}})^2 \sim \chi_{1,\alpha}^2
$$

where $\chi_{1,\alpha}^2$ is the upper $\alpha$ percentage point for the central [Chi-squared] with one d.f.  

This extends to the contingency table setting: whether the observed frequencies are equal to those expected under a null hypothesis of no association.  

### Test of Association
Pearson Chi-square test statistic is  

$$
\chi^2 = \sum_{\text{all categories}} \frac{(observed - epxected)^2}{expected}
$$

Comparison of proportions for several independent surveys or experiments  

| | Experiment 1 | Experiment 2 | ... | Experiment k |
|---|---|---|---|---|
| Number of successes | $x_1$| $x_2$ | ... | $x_k$ |
| Number of failures | $n_1 - x_1$ | $n_2 - x_2$ | ... | $n_k - x_k$ |
| | $n_1$ | $n_2$ | ... | $n_k$| 

$H_0: p_1 = p_2 = ... = p_k$ vs. the alternative that the null is not true (at least one pair are not equal).  

We estimate the common value of the probability of success on a single trial assuming $H_0$ is true:  

$$
\hat{p} = \frac{x_1 + x_2 + ... + x_k}{n_1 + n_2 + ...+ n_k}
$$


we use table of expected counts when $H_0$ is true:  

| | | | | |
|---|---|---|---|---|
|success | $n_1 \hat{p}$ | $n_2 \hat{p}$ | ... | $n_k \hat{p}$ |
|failure | $n_1(1-\hat{p})$ | $n_2(1-\hat{p})$ | ... | $n_k (1-\hat{p})$|
|| $n_1$ | $n_2$ | ... | $n_k$ |

$$
\chi^2 = \sum_{\text{all cells in table}} \frac{(observed - expected)^2}{expected}
$$

with k-1 degrees of freedom 


#### Two-way Count Data

| | 1 | 2 | ... | j | ... | c | Row Total|
|---|---|---|---|---|---|---|---|
|1|$n_{11}$ | $n_{12}$ | ... | $n_{1j}$| ... | $n_{1c}$ | $n_{1.}$|
|2|$n_{21}$ | $n_{22}$ | ... | $n_{2j}$| ... | $n_{2c}$ | $n_{2.}$|
| . | . | . | . | . | . | . | . |
|r|$n_{r1}$ | $n_{r2}$ | ... | $n_{rj}$| ... | $n_{rc}$ | $n_{r.}$|
|Column Total|$n_{.1}$ | $n_{.2}$ | ... | $n_{.j}$| ... | $n_{.c}$ | $n_{}$|

**Design 1**  
total sample size fixed n = constant (e.g., survey on job satisfaction and income); both row and column totals are random variables  

**Design 2**  
Fix the sample size in each group (in each row) (e.g., Drug treatments success or failure); fixed number of participants for each treatment; independent random samples from the two row populations.  

These different sampling designs imply two different probability models.  


#### Total Sample Size Fixed

**Design 1**

random sample of size n drawn from a single population, and sample units are cross-classified into r row categories and c column   

This results in an r x c table of observed counts 

$n_{ij} = 1,...,r;j=1,...,c$

Let $p_{ij}$ be the probability of classification into cell (i,j) and $\sum_{i=1}^r \sum_{j=1}^c p_{ij} = 1$. Let $N_{ij}$ be the random variable corresponding to $n_{ij}$  
The joint distribution of the $N_{ij}$ is multinomial with unknown parameters $p_{ij}$  

Denote the row variable by X and column variable by Y, then $p_{ij} = P(X=i,Y = j)$ and $p_{i.} = P(X = i)$ and $p_{.j} = P(Y = j)$ are the marginal probabilities.  
The null hypothesis that X and Y are statistically independent (i.e., no association) is just:  

$$
H_0: p_{ij} = P(X =i,Y=j) = P(X =i) P(Y =j) = p_{i.}p_{.j} \\
H_a: p_{ij} \neq p_{i.}p_{.j}
$$
for all i,j.  


#### Row Total Fixed 

**Design 2**

Random samples of sizes $n_1,...,n_r$ are drawn independently from $r \ge 2$ row populations. In this case, the 2-way table row totals are $n_{i.} = n_i$ for $i = 1,...,r$.  

The counts from each row are modeled by independent multinomial distributions.  

X is fixed, Y is observed.  

Then, $p_{ij}$ represent conditional probabilities $p_{ij} = P(Y=j|X=i)$  

The null hypothesis is the probability of response j is the same, regardless of the row population (i.e., no association):  

$$
H_0: p_{ij} = P(Y = j | X = i) = p_j \text{for all i,j =1,2,...,c} \\
\text{or } H_0: (p_{i1},p_{i2},...,p_{ic}) = (p_1,p_2,...,p_c) \text{ for all i} \\
H_a: (p_{i1},p_{i2},...,p_{ic}) \text{ are not the same for all i}
$$

Although the hypotheses to be tested are different for two sampling designs, **The chi-square test is identical**  

We have estimated expected frequencies:  

$$
\hat{e}_{ij} = \frac{n_{i.}n_{.j}}{n}
$$

The Chi-square statistic is  

$$
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(n_{ij}-\hat{e}_{ij})^2}{\hat{e}_{ij}} \sim \chi_{(r-1)(c-1)}
$$

$\alpha$-level test rejects $H_0$ if $\chi^2 > \chi^2_{(r-1)(c-1),\alpha}$

#### Pearson Chi-square Test

 * Determine whether an association exists  
 * Sometimes, $H_0$ represents the model whose validity is to be tested. Contrast this with the conventional formulation of $H_0$ as the hypothesis that is to be disproved. The goal in this case is not to disprove the model, but to see whether data are consistent with the model and if deviation can be attributed to chance.  
 * These tests do not measure the strength of an association.  
 * These tests depend on and reflect the sample size - double the sample size by copying each observation, double the $\chi^2$ statistic even thought the strength of the association does not change.  
 * The [Pearson Chi-square Test] is not appropriate when more than about 20% of the cells have an expected cell frequency of less than 5 (large-sample p-values not appropriate).  
 * When the sample size is small the exact p-values can be calculated (this is prohibitive for large samples); calculation of the exact p-values assumes that the column totals and row totals are fixed.  



```r
july.x=480 
july.n=1000 
sept.x=704 
sept.n=1600
```

$$
H_0: p_J = 0.5 \\
H_a: p_J < 0.5
$$


```r
prop.test(x=july.x,n=july.n,p=0.5,alternative="less",correct=F)
```

```
## 
## 	1-sample proportions test without continuity correction
## 
## data:  july.x out of july.n, null probability 0.5
## X-squared = 1.6, df = 1, p-value = 0.103
## alternative hypothesis: true p is less than 0.5
## 95 percent confidence interval:
##  0.0000000 0.5060055
## sample estimates:
##    p 
## 0.48
```


$$
H_0: p_J = p_S \\
H_a: p_j \neq p_S
$$


```r
prop.test(x=c(july.x,sept.x),n=c(july.n,sept.n),correct=F)
```

```
## 
## 	2-sample test for equality of proportions without continuity
## 	correction
## 
## data:  c(july.x, sept.x) out of c(july.n, sept.n)
## X-squared = 3.9701, df = 1, p-value = 0.04632
## alternative hypothesis: two.sided
## 95 percent confidence interval:
##  0.0006247187 0.0793752813
## sample estimates:
## prop 1 prop 2 
##   0.48   0.44
```




### Ordinal Association

 * An ordinal association implies that as one variable increases, the other tends to increase or decrease (depending on the nature of the association).  
 * For tests for variables with two or more levels, the levels must be in a logical ordering.  

#### Mantel-Haenszel Chi-square Test

The [Mantel-Haenszel Chi-square Test] is more powerful for testing ordinal associations, but does not test for the strength of the association.  

This test is presented in the case where one has a series of 2 x 2 tables that examine the same effects under different conditions (If there are K such tables, we have 2 x 2 x K table)  

In stratum k, given the marginal totals $(n_{.1k},n_{.2k},n_{1.k},n_{2.k})$, the sampling model for cell counts is the [Hypergeometric] (knowing $n_{11k}$ determines $(n_{12k},n_{21k},n_{22k})$, given the marginal totals)  

Assuming conditional independence, the [Hypergeometric] mean and variance of $n_{11k}$ are  

$$
m_{11k} = E(n_{11k}) = \frac{n_{1.k} n_{.1k}}{n_{..k}} \\
var(n_{11k}) = \frac{n_{1.k} n_{2.k} n_{.1k} n_{.2k}}{n_{..k}^2(n_{..k}-1)}
$$

To test conditional independence, Mantel and Haenszel proposed  

$$
M^2 = \frac{(|\sum_{k} n_{11k} - \sum_k m_{11k}| -.5)^2}{\sum_{k}var(n_{11k})} \sim \chi^2_{1}
$$
This method can be extended to general I x J x K tables. 

(2 x 2 x 3) table


```r
Bron=array(c(20, 9, 382, 214, 10, 7, 172, 120, 12, 6, 327, 183), dim = c(2, 2, 3), dimnames = list( Particulate = c("High", "Low"), Bronchitis = c("Yes", "No"), Age = c("15-24", "25-39", "40+"))) 
margin.table(Bron,c(1,2))
```

```
##            Bronchitis
## Particulate Yes  No
##        High  42 881
##        Low   22 517
```

```r
# assess whether the relationship between Bronchitis by Particulate Level varies by Age
library(samplesizeCMH)
marginal_table=margin.table(Bron,c(1,2))
odds.ratio(marginal_table)
```

```
## [1] 1.120318
```

```r
#  whether these odds vary by age. The conditional odds can be calculated using the original table.
apply(Bron,3,odds.ratio)
```

```
##     15-24     25-39       40+ 
## 1.2449098 0.9966777 1.1192661
```

```r
# Mantel-Haenszel Test
mantelhaen.test(Bron,correct=T)
```

```
## 
## 	Mantel-Haenszel chi-squared test with continuity correction
## 
## data:  Bron
## Mantel-Haenszel X-squared = 0.11442, df = 1, p-value = 0.7352
## alternative hypothesis: true common odds ratio is not equal to 1
## 95 percent confidence interval:
##  0.6693022 1.9265813
## sample estimates:
## common odds ratio 
##          1.135546
```


##### McNemar's Test 
special case of [Mantel-Haenszel Chi-square Test]


```r
vote=cbind(c(682,22),c(86,810))
mcnemar.test(vote,correct=T)
```

```
## 
## 	McNemar's Chi-squared test with continuity correction
## 
## data:  vote
## McNemar's chi-squared = 36.75, df = 1, p-value = 1.343e-09
```


#### Spearman Rank Correlation

To test for the strength of association between two ordinally scaled variables, we can use [Spearman Rank Correlation] statistic  

Let X and Y be two random variables measured on an ordinal scale. Consider n pairs of observations ($x_i,y_i$), i = 1,...,n  

The [Spearman Rank Correlation] coefficient (denoted by $r_S$ is calculated using the Pearson correlation formula, but based on the ranks of $x_i$ and $y_i$).  

[Spearman Rank Correlation] be calculated  

 1. Assign ranks to $x_i$'s and $y_i$'s separately. Let $u_i = rank(x_i)$ and $v_i = rank(y_i)$  
 2. Calculate $r_S$ using the formula for the Pearson correlation coefficient, but applied to the ranks: 

$$
r_S = \frac{\sum_{i=1}^{n}(u_i - \bar{u})(v_i - \bar{v})}{\sqrt{(\sum_{i = 1}^{n}(u_i - \bar{u})^2)(\sum_{i=1}^{n}(v_i - \bar{v})^2)}}
$$

$r_S$ ranges between -1 and +1 , with  

 * $r_S = -1$ if there is a perfect negative monotone association 
 * $r_S = +1$ if there is a perfect positive monotone association between X and Y.  

To test  

$H_0:$ X and Y independent  
$H_a$: X and Y positively associated  

For large n (e.g., $n \ge 10$), 

$$
r_S \sim N(0,1/(n-1))
$$

Then, 

$$
Z = r_s \sqrt{n-1} \sim N(0,1)
$$




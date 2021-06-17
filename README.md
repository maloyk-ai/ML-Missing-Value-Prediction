# Machine Learning - Predicting Missing Value using EM(Expectation Maximization

![missing1](https://user-images.githubusercontent.com/84564226/121764052-7974e380-cb5e-11eb-9fd5-96effc8292e5.jpeg)
Sometimes we have missing data, that is, variables whose values are unknown. For example, we have various sensors that are collecting statistics in realtime and some of which might fail at random or with a specific pattern to collect some or all of the statistical attributes from data. The corresponding data matrix will then have “holes” in it; these missing entries are often represented by NaN, which stands for “not a number”. The goal of imputation is to infer plausible values for the missing entries. Thisis sometimes called matrix completion. The ability to handle missing data in a principled way is one  of the biggest advantages of generative models.

To formalize our assumptions, we can associate a binary response variable r<sup>i</sup> ∈ {0, 1}, that specifies whether each value x<sup>i</sup> is observed or not. The joint model has the form p(x<sup>i</sup>, r<sup>i</sup>|θ, φ) = p(r<sup>i</sup>|x<sup>i</sup>, φ)p(x<sup>i</sup>|θ), where φ are the parameters controlling whether the item is observed or not. If we assume p(r<sup>i</sup>|x<sup>i</sup>, φ) = p(r<sup>i</sup>|φ), we say the data is missing completely at random or MCAR. If we assume p(r<sup>i</sup>|x<sup>i</sup>, φ) = p(r<sup>i</sup>|x<sub>o</sub><sup>i</sup> , φ), where x<sub>o</sub><sup>i</sup> is the observed part of x<sup>i</sup>, we say the data is missing at random or MAR. If neither of these assumptions hold, we saythe data is not missing at random or NMAR. In this case, we have to model the missing data mechanism, since the pattern of missingness is informative about the values of the missing dataand the corresponding parameters. Here, we will consider MAR.

Suppose we are missing some entries in a design matrix. If the columns are correlated, we canuse the observed entries to predict the missing entries. To reconstruct the scenario, We've sampled some data from a 3 dimensional Gaussian, and then deliberately “hid” for example 50% of the datain each row. We then inferred the missing entries given the observed entries, using the true(generating) model. More precisely, for each row i, we compute p(x<sub>h</sub><sup>i</sup>|x<sub>v</sub><sup>i</sup>, θ), where h<sup>i</sup> and o<sup>i</sup> are the indices of the hidden and visible entries in case i. From this, we compute the marginal distribution of each missing variable, p(x<sub>h</sub><sup>i</sup>j |x<sub>o</sub><sup>i</sup>, θ). 


EM is not equivalent to simply replacing variables by their expectations and applying the standard MLE formula; that would ignore the posterior variance and would result in an incorrect estimate. Instead we must compute the expectation of the sufficient statistics, and plug that into the usual equation for the MLE. We can easily modify the algorithm to perform MAP estimation.

**EM monotonically increases the observed data log likelihood**

EM monotonically increases the observed data log likelihood until it reaches a local optimum. As a consequence of this result, if you do not observe monotonic increase of the observeddata log likelihood, you must have an error in your math and/or code.


<p align="center">
  <img src="https://github.com/maloyk-ai/MachineLearning/blob/main/log_observed_likelihood.png" height="120%" width="120%" title="observed data log likelihood">
</p>

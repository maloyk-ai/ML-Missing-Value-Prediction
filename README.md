# Machine Learning - Predicting Missing Value using EM(Expectation Maximization)


![missing1](https://user-images.githubusercontent.com/84564226/121764052-7974e380-cb5e-11eb-9fd5-96effc8292e5.jpeg)


**EM monotonically increases the observed data log likelihood**
<p align="center">
  <img src="https://github.com/maloyk-ai/MachineLearning/blob/main/log_observed_likelihood.png" height="120%" width="120%" title="observed data log likelihood">
</p>

<p align="left">
<img src="https://github.com/maloyk-ai/MachineLearning/blob/main/imputed_parameter_1.png" height="75%" width="75%" title="Parameters">
<img src="https://github.com/maloyk-ai/MachineLearning/blob/main/imputed_parameter_2.png" height="75%" width="75%" title="Parameters">
<img src="https://github.com/maloyk-ai/MachineLearning/blob/main/imputed_parameters_3.png" height="75%" width="75%" title="Parameters">
</p>

<p style="font-family:'Courier New'" style="font-size:30px">
-------------------------------------------Initialization-------------------------------------------
<br/>
NaN Rate:0.200000<br/>
Actual NaN Rate: 0.20000<br/>
----------------------------------------------------------------------------------------------------
<br/>
NaN Rate: 0.2<br>
mu: [1 2 6]<br>
Sigma:<br>
 [[118  62  44]<br/>
 [ 62  49  17]<br/>
 [ 44  17  21]]<br>
----------------------------------------------------------------------------------------------------
 <br/>
Imputed mu: [1.242 2.129 6.092]<br>
Imputed Sigma:<br/>
 [[120.107  63.124  44.116]<br>
 [ 63.124  49.872  17.118]<br/>
 [ 44.116  17.118  20.948]]<br/>

----------------------------------------------------------------------------------------------------
|x_truth                   |        x_missing                    |       x_imputed                 |
|--------------------------|-------------------------------------|---------------------------------|
|[13.768 11.276 12.856]    |         [13.768 11.276    nan]      |      [9.761]                    |
|[2.275 4.083 6.369]       |         [2.275 4.083   nan]         |      [5.958]                    |
|[-16.557  -7.81   -0.28 ].|         [-16.557     nan     nan]   |      [-7.226 -0.446].           |
|[-3.529  0.393  3.63 ]    |         [-3.529  0.393    nan]      |      [4.059]                    |
|[11.018 -5.547 14.012]    |         [11.018    nan    nan]      |      [7.267 9.683]              |
|[-26.798 -24.544   2.028] |         [-26.798     nan   2.028]   |      [-20.583]                  |
|[-24.612 -13.364  -6.59 ] |         [    nan -13.364  -6.59 ]   |      [-29.336]                  |
|[-12.234  -5.829  -1.683] |         [   nan -5.829 -1.683]      |      [-16.342]                  |     
|[-2.083  4.289  4.207]    |         [-2.083  4.289    nan]      |      [3.45]                     |
|[-0.363 -4.435  8.031]    |         [-0.363    nan  8.031]      |      [-1.949]                   |
----------------------------------------------------------------------------------------------------

 </p>

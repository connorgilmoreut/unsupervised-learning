# Unsupervised Learning - Final Exam
## Big Data and Sufficient Statistics
Turn Big Data into Small Data:
* reducing column count (pea, factor analysis, etc.)
* reducing row count (mean, variance, correlation coefficient, cluster analysis) 

Nonparametric data problems
* not much data reduction is possible
* ordering the original data is the maximal information-preserving reduction that can be done

Factorization Theorem: 
* if the joint density of the data can be factored into a function g(T, theta) of T and theta alone and a function h(x1…x2)  of the data but not of theta. 

## Factorization Theorem
* The purpose of the Factorization Theorem is to provide an easy way to find a statistic that is sufficient
* Factorization Theorem factors the numerator of the conventional density function. The numerator and the denominator are proportional to each other, so they cancel out the ration, leaving only parts that do not depend upon theta. 

## PCA Part 1
* PCA seeks to find structure in the columns rather than in the rows

## Factor Analysis
* In estimating a factor, we should give more weight to manifest variables that feel the factor more strongly and less weight to manifest variables that feel the factor less strongly
	* we can infer the size of the unobservable correlation between a factor and a manifest variable from the observable correlation between manifest variables 
	* then we can calculate the weights to apply to the manifest variables to estimate the factors
* estimates of factors are linear combinations of the manifest variables 
* PCA vs FA
	* In PCA, the components (factors) are functions of the observable variables 
	* In FA, the observables are considered to be functions of the factors
	* In PCA, # of principal components = #
* structure matrix is a table of correlations between manifest variables and common factors
* The correlation between a manifest variable and a common factor is the slope of the regression of the manifest variable and the common factor
* The factorial determination of the variables is the degree to which the observed variables are determined by the common factors and can be measured by the average communality:
![](Screen%20Shot%202022-03-08%20at%202.13.46%20PM.png)
* Cannot test whether the FA model applies
	* appropriateness of the FA model can never be proven
	* the number of factors can never be known
	* the factor loadings can never be known even if the correlation matrix of the variables were known exactly and without sampling or measurement error
* most common factor extraction method 
	* uses the techniques of PCA
	* the correlation matrix of observed variables is analyzed into orthogonal linear combinations in such a way as to maximize the variance of each factor subject to the factor being uncorrelated with all preceding factors 
	* 


#unsupervised-learning

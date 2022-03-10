# Unsupervised Learning - Final Exam
## Big Data and Sufficient Statistics
Turn Big Data into Small Data:
* reducing column count (pca, factor analysis, etc.)
* reducing row count (mean, variance, correlation coefficient, cluster analysis) 

Nonparametric data problems
* not much data reduction is possible
* ordering the original data is the maximal information-preserving reduction that can be done

Factorization Theorem: 
* if the joint density of the data can be factored into a function g(T, theta) of T and theta alone and a function h(x1…x2)  of the data but not of theta. 

## Factorization Theorem
* The purpose of the Factorization Theorem is to provide an easy way to find a statistic that is sufficient
* Factorization Theorem factors the numerator of the conventional density function. The numerator and the denominator are proportional to each other, so they cancel out the ratio, leaving only parts that do not depend upon theta. 

## PCA Part 1
* PCA seeks to find structure in the columns rather than in the rows
* systems of linear equations are equivalent to a rotation of coordinate systems, which are equivalent to a change of perspective.
* How to tell if transformations maintain right angles with each other (are orthonormal)? 
	* If the cosine of the angle between each pair is zero
* All PCs are orthonormal, but not all orthonormal transformations are PCs
* Standardization means that the mean of each variable is subtracted from each coordinate and the result is then divided by the standard deviation of the variable
* the importance of a component and the magnitude of its variation are related to the length of its corresponding ellipsoid axis
* The correlation between an original variable and a principal component is called a loading. Analysts like to look at the loadings matrix to interpret component meanings.
* The principal components are actually the eigenvectors of the correlation matrix of the original (U,V,W) data.10 The squared lengths of the principal component axes are actually the eigenvalues of the correlation matrix of the original (U,V,W) data.

## Principal Components Analysis Part 2
* Thus the total variance = p if the variables have been standardized.
* Rules that have been proposed for discarding PRINs:
![](Screen%20Shot%202022-03-08%20at%2011.41.08%20PM.png)

* the eigenvalue for each PRIN is the variance of the PRIN
	* sum of eigenvalues = total variance
* eigenvectors are the coefficients of the principal components (used for interpretation)

## Figuring Out the Principal Component Coefficients and Why it Matters
## Principal Components Analysis for Two Variables
* eigenvalue 1 = 1 + r (r = corr(X1, X2))
* eigenvalue 2 = 1 - r (r = corr(X1,X2))
* there is one eigenvector for each eigenvalue 
* eigenvectors 
	* any vector is a solution if v1 = -v2
* the regression of X2 on X1 has slope r
* the regression of X1 on X2 has slope 1/r
* PC1 minimizes the total squared perpendicular distance from the points to the PC1 line

## Cluster Analysis 
* PCA vs Cluster Analysis
	* PCA actually can reduce the number of columns, CA does not reduce the number of rows
	* PCA changes the values of the variables, CA does not
	* In some cases, each cluster can be treated as if each were a single entity, so CA reduces the # of entities to understand
* Five basic steps in all CA studies:
	* select the observations to be clustered
	* define the variables to use in clustering the observations
	* compute similarities (or dissimilarities) among the observations
	* create groups (clusters) of similar observations
	* validate the resulting clusters
* quantifiable characteristics of clusters:
	* density
	* variance
	* dimension (size or radius)
	* shape (typically ellipsoidal) 
	* separation
* correlation and distance are two concepts that underlie most techniques for calculating (dis)similarities 
	* find correlation among rows
	* problems: high variations in smaller numbers and small variations in large numbers, standardization, using PCA or FA to prep data pre-clustering
* correlation measures (similarities)
	* profile: a row in a dataset
	* similarity between profiles (rows) has three parts:
		* shape (patterns of dips and rises across variables)
			* correlation is only sensitive to shape
		* scatter (dispersion of scores around their average)
		* elevation (mean score of the case over all variables)
	* correlation measures for similarities are seldom used in practice
* distance measures (dissimilarities) 
	* common distance measures:
![](Screen%20Shot%202022-03-09%20at%2012.20.51%20PM.png)
	* for Euclidean distance, you don’t need to take the square root because sum of squares is easy enough
	* mahalanobis distance generalizes the Euclidean distance to adjust for different variances and correlations between the variables
		* mahalanobis and euclidean distances are the same if the variables are uncorrelated and have the same variance 

### hierarchical agglomerative
* most-used clustering methods
* terminology
	* n steps
	* leaf = case
	* root = the super-cluster at the top (right sides) consisting of all the leaves
	* branch = cluster containing more than one, but not all, cases
	* node = a leaf, branch, or a root
	* a leaf is a node with no children, a root is a node with no parent
	* if every cluster has at most two children the tree is binary 
* degree of relatedness of two cases is indicated by their most recent common ancestor
* common hierarchical agglomerative methods
	* single linkage (nearest neighbor) 
		* uses the similarity (or least dissimilarity) between the entity and _any_ member of the cluster
	* complete linkage (farthest neighbor) 
		* like single linkage, except that similarity is the distance between the farthest members of the two entities. 
		* a joining entity must be close to all members of the entity to be joined
	* average linkage (average neighbor)
		* like complete linkage, except that the average level of similarity between any candidate entity for inclusion into an existing cluster and the individual members of that existing cluster must be within a specified level.
		* the similarity matrix must be recomputed after each step
		* helps to find clusters of roughly equal variance
	* ward’s method 
		* like single linkage, clusters are joined that result in the least increase in the within-cluster variance
		* within-cluster variance calculation:
			* Xikl = value of case l on variable k in cluster I 
			* Xbarik = the mean of all the nth variable values that are in cluster I
			* within-cluster variance for a cluster = (sum of squares of Xikl - Xbarik) / degrees of freedom
				* don’t necessarily need to divide by the degrees of freedom
			* sum across all clusters and then sum that across all variables 
		* Ward’s method minimizes within-cluster sum-of-squares and maximizes the between-cluster sum-of-squares at the same time
		* TSS = BSS + WSS
			* TSS is fixed
			* WSS and BSS have an inverse relationship
		* there is a trade-off between explanatory power (R-square) and parsimony (number of clusters)
	* centroid method
		* centroid = vector of means of the p variables of the cases in the cluster
		* like single linkage, except that the (dis)similarity between two clusters is defined to be the squared Euclidean distance between the two centroids of the two clusters 
	* density linkage
		* like single linkage, except that the (dis)similarity between two clusters is based on nonparametric density estimation 

### non-hierarchical methods: iterative partitioning  
* k-means clustering is the most well-known 
* steps
	1. begin by dividing the cases into a number of preliminary clusters 
	2. centroids become the seeds
	3. each case is assigned to the nearest seed
	4. new centroids / seeds calculated
	5. repeat until no changes in cluster membership
* 3 major decisions to make in iterative partitioning
	* initial partition 
	* type of pass (e.g. k-means, hill-climbing) 
	* statistical criterion (e.g. distance) 
* k-means clustering
	* minimize WSS 
	* resembles Ward’s method but in an iterative way

### factor analysis variants
* rotate data 90 degrees and apply FA
* Q-mode assumes cases are standardized to mean 0 and stdev 1
	* this ignores elevation and scatter in the profiles and groups only by their shapes

### hierarchical divisive
* logical opposites of hierarchical agglomerative methods
* begin with all cases in one cluster and divide it into successively smaller groups

### clumping methods
* permits overlapping clusters 
* cases can share membership in more than one cluster
* not in SAS

### graph theoretic methods
* builds on graph links
* not in SAS

### determining the number of clusters 
* some advice
	* pay attention to the underlying theory
	* three top performing heuristic methods
		* cubic clustering criterion (CCC)
			* local peaks
		* pseudo-F statistic (measures separation)
			* local peaks
		* pseudo-T2 statistic (measures separation b/w the two clusters most recently joined)
			* one more than the peak
	* vary the K parameter 
	* make a qualitative assessment
	* plot the data

### validating the cluster analysis
* two methods rejected by theoreticians / fail simulation studies
	* cophenetic correlation
		* hierarchical agglomerative methods
		* correlation between original similarity matrix and an artificial similarity matrix constructed from the tree
		* measures how well the tree actually represents the similarities between entities
	* M/ANOVA F-statistics
		* theoretically invalid 
* useful techniques
	* cross-validation 
		* run CA on first set of cases and use its centroids as seeds to cluster the second set
			* if different clusters appear in each set, then CA is not generalizable
	* try with new variables
		* like cross validation, bu split variables not cases
	* simulation 
		* create a simulated data set with major characteristics like those of real data
		* run CA on each and compare results

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
* most common factor extraction method (Method=Principal)
	* uses the techniques of PCA
	* the correlation matrix of observed variables is analyzed into orthogonal linear combinations in such a way as to maximize the variance of each factor subject to the factor being uncorrelated with all preceding factors 
	* commonalities equal to one is the same as PCA (no uniqueness factors)
	* priors other than one is principal factor analysis 
		* Squared Multiple Correlation 
* Iterated Principal method (Method=Print)
	* starts as PCA, then becomes PFA
	* repeats until change in commonalities is very minor or iteration limit is reached
* Maximum Likelihood (Method=ML)
	* number of common factors is assumed known 
	* priors = SMC
	* method estimates the factor loadings most likely to have produced the observed correlation matrix
	* iterative and time-consuming 
	* Heywood cases = estimated commonalities exceeding one
	* hypothesis test 1
		* H0 : no common factors
		* HA : at least one common factor 
	* hypothesis test 2
		* H0: the number of factors extracted is correct
		* HA: more factors are needed
* Unweighted Least Squares (Method=ULS)
	* number of factors is assumed known 
	* chooses factor loadings to minimize the squared differences between the observed X-correlations and the correlations resulting from the estimated factor loadings 
* Method=Alpha
	* cases are considered to be a population 
	* maximizes the correlation of the extracted factors with the corresponding population factors
* Method=Image
	* image of a variable is the part of it that is explainable by a linear combination of other variables
	* anti-image is the part that is not thus explainable 

### interpret the meaning of the factors
* similar to PCA
* examine the factor pattern (loadings) 
* if the initial factor pattern reveals no clear interpretation, a rotation may improve interpretability 
	* rotation
		* a shift in the angular orientation of the factor axes
		* a set of linear combinations of the factors, yielding new factors with new interpretations 
	* varimax: maximizes the variance of squared loadings for each factor, simplifying the columns
	* quartimax: maximizes the variance of squared loadings for each variable, simplifying the rows
	* equamax: maximizes a weighted average of the variance of squared loadings fore each factor and each variable, combining the above two 
	* for oblique rotations, the factors are now correlated.
		* reference axis correlations: partial correlations among pairs of factors with the effects of all other factors removed from the pair being correlated
		* reference structure: set of semi partial correlations between the variables and the common factors with the effects of all other common actors removed from the factor being correlated but not from the variable being correlated 
		
## Estimation of Factor Loadings and Factor Scores
* each manifest variable is a linear combination of p common factors and a single unique factor
* common factors are all uncorrelated with each other and standardized 
* unique factors are all uncorrelated with each other and with all common factors, but are only centered (mean zero)
* two steps
	* estimation of factor coefficients (factor loadings)
		* usually this can be enough
	* supply estimates of the factor scores 
* supplying estimates of the factor scores
	* display the formula for the regression of actual scores on manifest variables 
	* substitute the estimates from step one 
	* all that is needed is the formula for this regression (a function of the just-calculated factor loadings)

### estimation of factor coefficients 
* the correlation of two standardized variables is their covariance 
* R pxp 
	* loadings matrix * loadings matrix + diagonal matrix of variance of the unique factor (the uniqueness matrix) 
* Factor analysis estimates the loadings to make the theoretical correlations in Rpxp as close as possible to the empirically calculated correlations 
	* least-squares estimation to make the off-diagonal elements of the reduced correlation matrix as close as possible to the corresponding empirical correlations
	* getting the manifest correlations exactly correct is unusual and applies only rarely 
	* the number of off-diagonal terms in the correlation matrix grows in proportion to the square of the number of manifest variables 
* Least-squares estimation of factor coefficients 
	* main take-away: the implausible program of estimating factor coefficients when the factors cannot be observed can actually be carried out
* rotation
	* the solution for the correlation matrix Apxm is not unique 
	* we may rotate any initial factor solution by any orthonormal transformation that we choose in order to gain more insight and still reproduce the observed correlations as closely as the original solution 

### estimation of factor scores
* factor scores are calculated for individual observations 
* only necessary when desired
* regression of the factors on the manifest variables and predict the factor scores by plugging the observed values of the manifest variables into the equation
	* can’t do this because we don’t have the factor scores
* can be written in terms of lambdas that we estimated
* we want the coefficients from regressing the factors on the X variables
* since the factors and X variables are assumed to be standardized, then the regression intercepts are all zeroes

### summary
* estimation of loadings can be done separately from scores for two reasons:
	* the sample correlations between manifest variables can be empirically computed
	* the theoretical correlation between manifest variables is a function only of the loadings

## More on Estimation of Factor Scores
* the special case of factoring and no uniqueness shows that the estimation of factor scores is feasible 
	* only requires knowing the eigenvalues of the correlation matrix (lambda i) and the estimated factor coefficients (lambda ij) from least squares estimation

## Notes from Quizzes
* The R-square between a manifest variable and its unique factor is the uniqueness of the manifest = 1 - (0.8) * (0.8) = 0.36. Since the R-square is the square of the correlation, then correlation = sqrt(0.36) = 0.6.
* PCA coefficient = correlation / sqrt(eigenvalue of component)
* The original variables are standardized, but variance of component = eigenvalue of component.
* The correlation between manifest variables is the inner product of their factor coefficient vectors = (0.8) * (0.6) = 0.48.
* varimax, equamax, and quartimax are orthonormal transformations
* The correct formula for slope is b = corrr(Y,X) stdev(Y) / stdev(X).
* At the root, there is only one cluster, so there is no between SS, and BSS = 0, so R-square = 0.
* WSS measures within-cluster variability - how far the data in one cluster are from the cluster centroid + the same measure for all other clusters.  
* Ward’s method joins the two clusters that result in the least increase in WSS (the “error” SS). Since BSS + WSS = TSS (a constant throughout all steps), then BSS must also change (decrease) the least.
* There is only one PCA for two variables,
* PCA: The order of importance is in the order of the magnitudes of the coefficients (or correlation coefficients, since coefficients and correlations are proportional to each other.)
* The square root is a one-to-one function of its positive argument and so has the same information as the radicand (which is sufficient), and so is sufficient as well.
* Z1 is standardized. So its standard deviation is 1. Hence, its variance is 1.

#unsupervised-learning

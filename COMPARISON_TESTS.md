# R2 vs R — Side-by-Side Comparison Tests

Run R2 code in R2 console, R code in R console.
R2 code comes FIRST, R code comes SECOND.
R2 functions are built-in. R often needs install.packages() first.

#==========================================================================
# Comparison 1: Basic Statistics
# Both R and R2 have these in base. Results should match exactly.
#==========================================================================

#--- R2 code ---
x <- c(23, 45, 12, 67, 34, 89, 56, 78, 90, 11)
mean(x)
sd(x)
median(x)
var(x)
range(x)
quantile(x, 0.25)
IQR(x)

#--- R code (no library needed) ---
x <- c(23, 45, 12, 67, 34, 89, 56, 78, 90, 11)
mean(x)
sd(x)
median(x)
var(x)
range(x)
quantile(x, 0.25)
IQR(x)

#==========================================================================
# Comparison 2: Linear Regression
# Both have lm() in base. R2 summary matches R with stars.
#==========================================================================

#--- R2 code ---
model <- lm(mpg ~ wt + hp, data = mtcars)
summary(model)
coef(model)

#--- R code (no library needed) ---
model <- lm(mpg ~ wt + hp, data = mtcars)
summary(model)
coef(model)

#==========================================================================
# Comparison 3: Linear Regression Speed (10K rows, 10 predictors)
# Both should complete in ~0.01s
#==========================================================================

#--- R2 code ---
set.seed(42)
x <- matrix(rnorm(100000), nrow = 10000, ncol = 10)
y <- rnorm(10000)
system.time(lm(y ~ x))

#--- R code (no library needed) ---
set.seed(42)
x <- matrix(rnorm(100000), nrow = 10000, ncol = 10)
y <- rnorm(10000)
system.time(lm(y ~ x))

#==========================================================================
# Comparison 4: Matrix Multiply Speed (1000x1000)
# R2 ~0.24s, R ~0.54s. R2 is 2.2x faster.
#==========================================================================

#--- R2 code ---
x <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
system.time(t(x) %*% x)

#--- R code (no library needed) ---
x <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
system.time(t(x) %*% x)

#==========================================================================
# Comparison 5: Random Forest Speed (10K rows, 100 trees)
# R2 ~0.5s (6-core parallel), R ~1.2s (single core)
# R needs randomForest package. R2 built-in.
#==========================================================================

#--- R2 code ---
set.seed(42)
x_med <- matrix(runif(100000), nrow = 10000, ncol = 10)
y_med <- c(rep(1, 5000), rep(2, 5000))
system.time(rf(x_med, y_med, ntrees = 100))

#--- R code (NEEDS LIBRARY) ---
if (!require("randomForest")) install.packages("randomForest")
library(randomForest)
set.seed(42)
x_med <- matrix(runif(100000), nrow = 10000, ncol = 10)
y_med <- as.factor(c(rep(1, 5000), rep(2, 5000)))
system.time(randomForest(x_med, y_med, ntree = 100))

#==========================================================================
# Comparison 6: Decision Tree
# R needs rpart package. R2 built-in.
#==========================================================================

#--- R2 code ---
x <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow = 150, ncol = 4)
y <- c(rep(1, 50), rep(2, 50), rep(3, 50))
rpart(x, y)

#--- R code (NEEDS LIBRARY) ---
if (!require("rpart")) install.packages("rpart")
library(rpart)
rpart(Species ~ ., data = iris)

#==========================================================================
# Comparison 7: Gradient Boosting
# R needs gbm package + different parameter names. R2 built-in.
#==========================================================================

#--- R2 code ---
g <- gbm(mpg ~ ., data = mtcars, ntrees = 100, learning_rate = 0.1)
summary(g)

#--- R code (NEEDS LIBRARY) ---
if (!require("gbm")) install.packages("gbm")
library(gbm)
g <- gbm(mpg ~ ., data = mtcars, n.trees = 100, shrinkage = 0.1, distribution = "gaussian")
summary(g)

#==========================================================================
# Comparison 8: K-means Clustering
# Both have kmeans() in base. Same syntax.
#==========================================================================

#--- R2 code ---
set.seed(42)
x <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow = 150, ncol = 4)
km <- kmeans(x, centers = 3)
summary(km)

#--- R code (no library needed) ---
set.seed(42)
x <- as.matrix(iris[, 1:4])
km <- kmeans(x, centers = 3)
km

#==========================================================================
# Comparison 9: PCA
# Both have prcomp() in base. PC1 sdev ~2.05.
#==========================================================================

#--- R2 code ---
x <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow = 150, ncol = 4)
prcomp(x)

#--- R code (no library needed) ---
prcomp(iris[, 1:4])

#==========================================================================
# Comparison 10: T-test
# Both have t.test() in base. Same syntax.
#==========================================================================

#--- R2 code ---
x <- c(5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9)
t.test(x, mu = 5)

#--- R code (no library needed) ---
x <- c(5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9)
t.test(x, mu = 5)

#==========================================================================
# Comparison 11: Chi-squared Test
# Both in base. R2 auto-applies Yates' correction on 2x2 (like R).
#==========================================================================

#--- R2 code ---
chisq.test(c(10, 20, 30))
chisq.test(matrix(c(762, 327, 468, 484), nrow = 2))

#--- R code (no library needed) ---
chisq.test(c(10, 20, 30))
chisq.test(matrix(c(762, 327, 468, 484), nrow = 2))

#==========================================================================
# Comparison 12: ANOVA
# Both in base. F = 119.26, p < 2e-16.
#==========================================================================

#--- R2 code ---
aov(Sepal.Length ~ Species, data = iris)

#--- R code (no library needed) ---
summary(aov(Sepal.Length ~ Species, data = iris))

#==========================================================================
# Comparison 13: Shapiro-Wilk Normality Test
# Both in base. W ~0.976, p ~0.01.
#==========================================================================

#--- R2 code ---
shapiro.test(iris$Sepal.Length)

#--- R code (no library needed) ---
shapiro.test(iris$Sepal.Length)

#==========================================================================
# Comparison 14: Correlation Test
# Both in base. r ~0.87, p < 2e-16.
#==========================================================================

#--- R2 code ---
cor.test(iris$Sepal.Length, iris$Petal.Length)

#--- R code (no library needed) ---
cor.test(iris$Sepal.Length, iris$Petal.Length)

#==========================================================================
# Comparison 15: Cross-Validation
# R needs caret package + 4 lines setup. R2 one line.
#==========================================================================

#--- R2 code ---
mat <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow = 150, ncol = 4)
y <- iris$Petal.Length
cv(mat, y, model = "lm", k = 5)

#--- R code (NEEDS LIBRARY) ---
if (!require("caret")) install.packages("caret")
library(caret)
ctrl <- trainControl(method = "cv", number = 5)
model <- train(Petal.Length ~ ., data = iris, method = "lm", trControl = ctrl)
model$results

#==========================================================================
# Comparison 16: Confusion Matrix
# R needs caret package. R2 built-in with precision/recall/F1.
#==========================================================================

#--- R2 code ---
pred <- c(rep(1,48), rep(2,2), rep(1,5), rep(2,45), rep(2,3), rep(3,47))
actual <- c(rep(1,50), rep(2,50), rep(3,50))
confusion.matrix(pred, actual)

#--- R code (NEEDS LIBRARY) ---
if (!require("caret")) install.packages("caret")
library(caret)
pred <- factor(c(rep(1,48), rep(2,2), rep(1,5), rep(2,45), rep(2,3), rep(3,47)))
actual <- factor(c(rep(1,50), rep(2,50), rep(3,50)))
confusionMatrix(pred, actual)

#==========================================================================
# Comparison 17: Data Frame Operations
# R2 has pipe-friendly filter/select. R uses bracket notation.
#==========================================================================

#--- R2 code ---
head(iris, 5)
summary(iris)
filter(iris, iris$Sepal.Length > 7)
select(iris, "Sepal.Length", "Species")

#--- R code (no library needed, or dplyr for filter/select) ---
head(iris, 5)
summary(iris)
iris[iris$Sepal.Length > 7, ]
iris[, c("Sepal.Length", "Species")]

#==========================================================================
# Comparison 18: CSV Read/Write
# Same syntax in both. R2 auto-detects column types.
#==========================================================================

#--- R2 code ---
write.csv(iris, "test_iris.csv")
d <- read.csv("test_iris.csv")
head(d)
str(d)

#--- R code (no library needed) ---
write.csv(iris, "test_iris.csv")
d <- read.csv("test_iris.csv")
head(d)
str(d)

#==========================================================================
# Comparison 19: Save/Load Models
# R2 uses .r2m with class auto-detection. R uses .rds.
#==========================================================================

#--- R2 code ---
g <- gbm(mpg ~ ., data = mtcars, ntrees = 50)
save(g, "model.r2m")
m <- load("model.r2m")
class(m)
summary(m)

#--- R code (NEEDS LIBRARY) ---
if (!require("gbm")) install.packages("gbm")
library(gbm)
g <- gbm(mpg ~ ., data = mtcars, n.trees = 50, distribution = "gaussian")
saveRDS(g, "model.rds")
m <- readRDS("model.rds")
class(m)
summary(m)

#==========================================================================
# Comparison 20: System Info
#==========================================================================

#--- R2 code ---
version()

#--- R code (no library needed) ---
R.version
sessionInfo()

#==========================================================================
# RESULTS SUMMARY
#==========================================================================
#
# | Test                  | R             | R2            | R2 Advantage  |
# |-----------------------|---------------|---------------|---------------|
# | Install size          | 200+ MB       | 5 MB          | 40x smaller   |
# | Matrix 1000x1000      | 0.54s         | 0.24s         | 2.2x faster   |
# | Random Forest 10K     | 1.21s         | 0.53s         | 2.3x faster   |
# | lm() 10K rows         | 0.01s         | 0.01s         | Equal         |
# | Decision tree         | needs rpart   | built-in      | Zero install  |
# | Random forest         | needs pkg     | built-in      | Zero install  |
# | Gradient boosting     | needs gbm     | built-in      | Zero install  |
# | Cross-validation      | needs caret   | built-in      | Zero install  |
# | Confusion matrix      | needs caret   | built-in      | Zero install  |
# | Total built-in funcs  | ~400 base     | 192 built-in  | All-in-one    |
# | External dependencies | hundreds      | Rust-only     | No C/C++      |
#

# R2 Test Cases — Verification Suite

Run these in R2 to verify all features work correctly.

## Test 1: Basic Language
```r
x <- c(1,2,3,4,5)
mean(x)                    # Expected: 3
x[2:4]                     # Expected: 2 3 4
length(x)                  # Expected: 5
```

## Test 2: Data Frames
```r
head(iris, 5)
summary(iris)
iris[iris$Sepal.Length > 7, ]
str(iris)
```

## Test 3: Linear Regression
```r
model <- lm(mpg ~ wt + hp, data = mtcars)
summary(model)
coef(model)
plot(model)                # Creates plot.svg
```

## Test 4: Decision Tree
```r
rpart(Petal.Length ~ ., data = iris)
```

## Test 5: Random Forest
```r
rf(Petal.Length ~ ., data = iris, ntrees = 50)
```

## Test 6: Gradient Boosting
```r
g <- gbm(mpg ~ ., data = mtcars, ntrees = 100)
summary(g)
plot(g)                    # Creates plot.svg (loss curve)
```

## Test 7: Clustering
```r
km <- kmeans(matrix(rnorm(300), 100, 3), centers = 3)
summary(km)
```

## Test 8: CSV Read/Write
```r
write.csv(iris, "test.csv")
d <- read.csv("test.csv")
head(d)
```

## Test 9: Save/Load
```r
# Session
x <- 42
save("session.r2s")
rm("x")
load("session.r2s")
x                          # Expected: 42

# Model
g <- gbm(mpg ~ ., data = mtcars, ntrees = 50)
save(g, "model.r2m")
m <- load("model.r2m")
class(m)                   # Expected: "gbm"
summary(m)

# Data
save(iris, "iris.r2d")
d <- load("iris.r2d")
head(d)
```

## Test 10: Help System
```r
?lm
?gbm
??kmeans
help()
version()
```

## Test 11: Benchmark (compare with R)
```r
x <- matrix(rnorm(1000000), nrow = 1000, ncol = 1000)
system.time(t(x) %*% x)

set.seed(42)
x <- matrix(rnorm(100000), nrow = 10000, ncol = 10)
y <- rnorm(10000)
system.time(lm(y ~ x))
```

## Test 12: String Operations
```r
x <- c("hello world", "foo bar", "hello R2")
grepl("hello", x)          # Expected: TRUE FALSE TRUE
sub("hello", "HI", x)      # Expected: "HI world" "foo bar" "HI R2"
nchar(x)                   # Expected: 11 7 8
```

## Test 13: Data Manipulation
```r
filter(iris, iris$Sepal.Length > 7)
select(iris, "Sepal.Length", "Species")
order(c(3, 1, 4, 1, 5))    # Expected: 2 4 1 3 5
duplicated(c(1,2,3,2,1))   # Expected: FALSE FALSE FALSE TRUE TRUE
```

## Test 14: Cross-Validation
```r
mat <- matrix(c(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length, iris$Petal.Width), nrow=150, ncol=4)
y <- iris$Petal.Length
cv(mat, y, model = "lm", k = 5)
```

## Test 15: Confusion Matrix
```r
pred <- c(rep(1,48), rep(2,2), rep(1,5), rep(2,45), rep(2,3), rep(3,47))
actual <- c(rep(1,50), rep(2,50), rep(3,50))
confusion.matrix(pred, actual)
```

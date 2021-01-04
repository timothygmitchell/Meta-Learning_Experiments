tune_params <- function(k = 3, c = 2, g = 10, df) {
  
  # error handling
  if (class(df) != "data.frame") stop("df needs to be a data.frame")
  if ((k < 2) | (k > 10)) stop("k must be between 2 and 10")
  if (c < 2) stop("c must be at least 2")
  if (g < 2) stop("g must be at least 2")
  
  ### Parameter Tuning: XGBoost
  
  ptm <- begin_process("XGBoost")
  
  xgb.DMatrix <- xgb.DMatrix(sparse.model.matrix(Y ~ 0 + ., data = df), 
                             label = as.numeric(df$Y) - 1)
  
  searchGrid <- expand.grid(subsample = c(0.40, 0.55, 0.70, 0.85, 1.0),
                            colsample_bytree = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                            colsample_bynode = c(0.5, 0.75, 1),
                            max_depth = c(4, 6, 8, 10, 12, 14, 16),
                            eta = c(0.001, 0.01, .1, 0.3))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g*3)
  
  cv_error <- foreach (i = 1:c, .combine = 'list', .multicombine = TRUE) %do% {
    
    cv_error <- t(apply(searchGrid, 1, function(params) {
      
      cv_log <- xgb.cv(data = xgb.DMatrix,                               # training sample
                       nround = 5000,                                    # maximum number of trees
                       early_stopping_rounds = 20,                       # stopping threshold if no improvement
                       objective = "binary:logistic",                    # objective function
                       eval_metric = "error",                            # evaluation metric
                       maximize = FALSE,                                 # want to MINIMIZE error
                       max.depth = params[["max_depth"]],                # tree depth
                       eta = params[["eta"]],                            # learning rate
                       gamma = 0,                                        # minimum loss reduction
                       subsample = params[["subsample"]],                # sample fraction of original data
                       colsample_bytree = params[["colsample_bytree"]],  # how many features sampled, each tree
                       colsample_bynode = params[["colsample_bynode"]],  # how many features sampled, each node
                       verbose = FALSE,
                       showsd = FALSE,
                       nfold = k,                                        # number of cv folds
                       stratified = TRUE)                                # stratify folds to balance classes
      
      best_error <- min(cv_log$evaluation_log[ , test_error_mean])
      best_rounds <- match(best_error, cv_log$evaluation_log[ , test_error_mean])
      
      return(c("error" = best_error, "trees" = best_rounds, params))
      
    }))
    
    return(cv_error)
  }
  
  cv_error <- apply(simplify2array(cv_error), 1:2, mean)    # mean error across all cycles
  
  xgb.opt <- cv_error[order(cv_error[ , 1])[1], ]           # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Extremely Randomized Trees
  ptm <- begin_process("Extremely Randomized Trees")
  
  searchGrid <- expand.grid("mtry" = unique(floor(seq(5, ncol(df) - 1, length.out = 15))), 
                            "max.depth" = c(4, seq(5, 50, by = 5)))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)
  
  # k-fold cross-validation
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- df[fold != i, ]
      test <- df[fold == i, -1]
      
      mapply(function(x, y) {
        
        model <- ranger(Y ~ . , 
                        splitrule = "extratrees", 
                        replace = F, 
                        sample.fraction = 1,
                        data = train, 
                        num.trees = 300, 
                        mtry = x, 
                        max.depth = y)
        
        Kappa(predict(model, test)$predictions, df$Y[fold == i])
        
      }, x = searchGrid$mtry, y = searchGrid$max.depth)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = kfcv())
  
  xt.opt <- cv_error[which.max(cv_error$kappa), ]         # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Random Forest
  ptm <- begin_process("Random Forest")
  
  searchGrid <- expand.grid("mtry" = unique(floor(seq(5, ncol(df) - 1, length.out = 15))), 
                            "max.depth" = c(4, seq(5, 50, by = 5)))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)
  
  # k-fold cross-validation
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- df[fold != i, ]
      test <- df[fold == i, -1]
      
      mapply(function(x, y) {
        
        model <- ranger(Y ~ . , 
                        data = train, 
                        num.trees = 300, 
                        mtry = x, 
                        max.depth = y)
        
        Kappa(predict(model, test)$predictions, df$Y[fold == i])
        
      }, x = searchGrid$mtry, y = searchGrid$max.depth)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = kfcv())
  
  rf.opt <- cv_error[which.max(cv_error$kappa), ]         # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Elastic Net
  ptm <- begin_process("Regularized Regression (Elastic Net)")
  
  alpha <- seq(0, 1, by = 0.025) # elastic net mixing parameter
  
  alpha <- sample(alpha, min(g, length(alpha)))
  
  # function to find best shrinkage parameter (learning rate) for each value of alpha
  best_lambda <- function(alpha) {
    
    # glmnet standardizes variables internally
    cv.glmnet <- cv.glmnet(ohe(df), 
                           df$Y,
                           type.measure = "deviance",
                           alpha = alpha, 
                           nfolds = max(k, 3),
                           family = "binomial")
    
    return(c(alpha = alpha,
             lambda = cv.glmnet$lambda.min, 
             deviance = min(cv.glmnet$cvm)))
    
  }
  
  cv_error <- replicate(c, do.call(rbind, mclapply(alpha, function(x) {
    
    best_lambda(x)
    
  })), simplify = FALSE)
  
  cv_error <- data.frame(Reduce("+", cv_error)/c)        # average across c cycles
  
  en.opt <- cv_error[which.min(cv_error$deviance), ]     # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: k-Nearest Neighbors
  ptm <- begin_process("k-Nearest Neighbors")
  
  neighbors <- min(g*3, 40, floor(((k-1)/k)*nrow(df)) - 1)
  
  neighbors <- sort(sample(neighbors)[1:min(g*3, 40)])
  
  neighbors <- neighbors[!is.na(neighbors)]
  
  # k-fold cross-validation
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- ohe(standardize(df))[fold != i, ]
      test <- ohe(standardize(df))[fold == i, ]
      
      unlist(mclapply(neighbors, function(x) {Kappa(knn(train, test, 
                                                        cl = df$Y[fold != i], k = x),
                                                    df$Y[fold == i])}, 
                      mc.cores = detectCores() - 1))
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- data.frame(neighbors, kappa = rowMeans(replicate(c, kfcv()))) 
  
  head(cv_error[rev(order(cv_error$kappa)), ])       # minimum error
  
  knn.opt <- cv_error[which.max(cv_error$kappa), ]   # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  # Parameter Tuning: Neural Network
  ptm <- begin_process("Neural Network")
  
  searchGrid <- expand.grid(hidden = floor(seq(0.25*(ncol(ohe(df))), 
                                               1.35*(ncol(ohe(df))), 
                                               by = 1)), 
                            algorithm = c("rprop+", "rprop-"))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], g)
  
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- ohe.data.frame(standardize(df))[fold != i, ]
      
      test <- ohe.data.frame(standardize(df))[fold == i, -1]
      
      mapply(function(x, y) {
        
        nn <- neuralnet(Y ~ . , 
                        data = train, 
                        hidden = x, 
                        algorithm = y, 
                        rep = 1,
                        stepmax = 1e4,
                        threshold = 0.3,
                        linear.output = FALSE)
        
        Kappa(factor(round(predict(nn, test)[ , 2])), df$Y[fold == i])
        
      }, x = searchGrid$hidden, y = searchGrid$algorithm)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv()))
  
  head(cv_error[rev(order(cv_error$kappa)), ])        # minimum error
  
  nn.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Support Vector Machine (Linear Kernel)
  ptm <- begin_process("Support Vector Machine with Linear Kernel")
  
  searchGrid <- expand.grid(kernel = "linear", 
                            cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.05, 0.08, 0.1, 1, 5),
                            epsilon = c(0.001, 0.01, 0.1, 0.5))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], max(4, floor(g*.3)))
  
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- df[fold != i, ] 
      test <- df[fold == i, -1]
      
      mapply(function(x, y, z) {
        
        model <- svm(Y ~ .,
                     data = train,
                     kernel = x,
                     cost = y, 
                     epsilon = z)
        
        Kappa(factor(predict(model, test)), df$Y[fold == i])
        
      }, x = searchGrid$kernel, y = searchGrid$cost, z = searchGrid$epsilon)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv()))
  
  head(cv_error[rev(order(cv_error$kappa)), ])          # minimum error
  
  svml.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Support Vector Machine (Polynomial Kernel)
  ptm <- begin_process("Support Vector Machine with Polynomial Kernel")
  
  searchGrid <- expand.grid(kernel = "polynomial", 
                            degree = 1:3, 
                            coef0 = c(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
                            cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1),
                            epsilon = c(0.001, 0.01, 0.1))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], max(5, floor(g*.6)))
  
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- df[fold != i, ] 
      test <- df[fold == i, -1]
      
      mapply(function(v, w, x, y, z) {
        
        model <- svm(Y ~ .,
                     data = train,
                     kernel = v,
                     degree = w, 
                     coef0 = x,
                     cost = y,
                     epsilon = z)
        
        Kappa(factor(predict(model, test)), df$Y[fold == i])
        
      }, v = searchGrid$kernel, w = searchGrid$degree, x = searchGrid$coef0, 
      y = searchGrid$cost, z = searchGrid$epsilon)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv()))
  
  head(cv_error[rev(order(cv_error$kappa)), ])               # minimum error
  
  svmp.opt <- cv_error[which.max(cv_error$kappa), ]          # optimal hyperparameters
  
  end_process(ptm)
  
  #########################################################################################
  
  ### Parameter Tuning: Support Vector Machine (Radial Kernel)
  ptm <- begin_process("Support Vector Machine with Radial Kernel")
  
  searchGrid <- expand.grid(kernel = "radial", 
                            cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.05, 0.08, 0.1, 1, 5),
                            gamma = c(0.001, 0.005, 0.1, 0.5, 1, 2, 3, 4))
  
  searchGrid <- head(searchGrid[sample(1:nrow(searchGrid), nrow(searchGrid)), ], max(3, floor(g*.1)))
  
  kfcv <- function(i = k) {
    
    fold <- sample(cut(1:nrow(df), breaks = i, labels = FALSE))
    
    cv_error <- foreach (i = 1:k, .combine = 'cbind') %do% {
      
      train <- df[fold != i, ] 
      test <- df[fold == i, -1]
      
      mapply(function(x, y, z) {
        
        model <- svm(Y ~ .,
                     data = train,
                     kernel = x,
                     cost = y, 
                     gamma = z)
        
        Kappa(factor(predict(model, test)), df$Y[fold == i])
        
      }, x = searchGrid$kernel, y = searchGrid$cost, z = searchGrid$gamma)
      
    }
    
    return(rowMeans(cv_error))
    
  }
  
  cv_error <- cbind(searchGrid, kappa = cv_parallel(c, kfcv()))
  
  head(cv_error[rev(order(cv_error$kappa)), ])          # minimum error
  
  svmr.opt <- cv_error[which.max(cv_error$kappa), ]     # optimal hyperparameters
  
  # best kernel (radial, linear, polynomial)
  
  svm.list <- list(svmr.opt, svml.opt, svmp.opt)
  
  svm.opt <- svm.list[[which.max(sapply(svm.list, `[`, c("kappa")))]]
  
  svm.opt # best support vector machine in the discovery set
  
  end_process(ptm)
  
  cat("Writing hyperparameters to the global environment\n")
  
  # best hyperparameters 
  best_params <- list("xgb.opt" = xgb.opt, "xt.opt" = xt.opt, "rf.opt" = rf.opt, 
                      "en.opt" = en.opt, "knn.opt" = knn.opt, "nn.opt" = nn.opt, 
                      "svm.opt" = svm.opt)

  list2env(best_params, envir = .GlobalEnv)
  
}

tune_params(k = 2, c = 2, g = 10, df = "cat") # error
tune_params(k = 1, c = 2, g = 10, df = df)    # error
tune_params(k = 11, c = 2, g = 10, df = df)   # error
tune_params(k = 2, c = 2, g = 1, df = df)     # error
tune_params(k = 2, c = 1, g = 1, df = df)     # error

# save to directory

path <- "~/Documents/Github/Empirical_Sudy_of_Ensemble_Learning_Methods/utils/"

saveRDS(tune_params, file = paste0(path, "tune_params"))

# grab some hyperparameter quick-load values from a Higgs boson experiment
params <- list("xgb.opt" = xgb.opt, # XGBoost
               "xt.opt"  =  xt.opt, # extremely randomized trees
               "rf.opt"  =  rf.opt, # random forest
               "knn.opt" = knn.opt, # k-nearest neighbors
               "en.opt"  =  en.opt, # elastic net params
               "nn.opt"  =  nn.opt, # neural network
               "svm.opt" = svm.opt) # support vector machine

sapply(1:length(params), function(x) {
  
  saveRDS(params[[x]], file = paste0(path, names(params)[[x]]))
  return(paste("saved", names(params)[[x]]))
  
})

# now these can be easily retrieved with

params <- list("xgb.opt", # XGBoost
               "xt.opt",  # extremely randomized trees
               "rf.opt",  # random forest
               "knn.opt", # k-nearest neighbors
               "en.opt",  # elastic net params
               "nn.opt",  # neural network
               "svm.opt") # support vector machine

for (i in 1:length(params)) { assign(params[i], readRDS(paste0(path, params[i]))) }
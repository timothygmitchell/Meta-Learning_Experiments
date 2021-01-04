rm(list = ls())

# helper functions
cv_parallel <- function(n, expr, ...) {
  
  # function to perform parallel repeated k-fold cross-validation
  # the input is a function called kfcv() which I defined for each model
  rowMeans(simplify2array(mclapply(
    integer(n), eval.parent(substitute(function(...) expr)), ...)))
}

# bookkeeping functions to time a parameter tuning process and print the elapsed time to the console
begin_process <- function(model_name) {cat(paste0("Tuning ", model_name, "\n")); return(proc.time())}
end_process <- function(ptm) {cat(paste(round((proc.time() - ptm)[3], 2), "s elapsed\n"))}

# Cohen's Kappa: a loss function appropriate for imbalanced classes
Kappa <- function(pred, actual) {confusionMatrix(pred, actual)[[3]][2]}

# Matthews Correlation Coefficient: another loss function for imbalanced classes
MCC <- function (pred, actual) {
  
  CM <- caret::confusionMatrix(pred, actual)
  
  TP <- CM$table[1,1]; TN <- CM$table[2,2]
  FP <- CM$table[1,2]; FN <- CM$table[2,1]
  
  mcc_num <- (TP*TN - FP*FN)
  mcc_den <- as.double((TP+FP))*as.double((TP+FN))*as.double((TN+FP))*as.double((TN+FN))
  mcc <- mcc_num/sqrt(mcc_den)
  
  return(mcc)
}

# convert any binary outcome to a factor with levels 0 and 1
binary <- function(vec) {factor(ifelse(vec == unique(vec)[1], 1, 0))}

# center and scale numeric variables (and leave factors as is)
standardize <- function(df) {rapply(df, scale, c("numeric", "integer"), how = "replace")}

# convert to a design matrix with one-hot-encoding
ohe <- function(df) {model.matrix(Y ~ 0 + ., df)}

# convert to a design matrix with one-hot-encoding (dataframe output, class label included)
ohe.data.frame <- function(df) {data.frame(Y = df$Y, model.matrix(Y ~ 0 + ., df))}

phon <- "https://raw.githubusercontent.com/jbrownlee/Datasets/master/phoneme.csv"
phon <- read.csv(phon)[ , c(6, 1:5)]
colnames(phon) <- c("Y", paste0("X", 1:5))
phon$Y <- binary(phon$Y)

spam <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam <- read.csv(spam)[ , c(58, 1:57)]
colnames(spam) <- c("Y", paste0("X", 1:57))
spam$Y <- binary(spam$Y)

wdbc <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
wdbc <- read.csv(wdbc)[ , -1]
colnames(wdbc) <- c("Y", paste0("X", 1:30))
wdbc$Y <- binary(wdbc$Y)

adult <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult <- read.csv(adult, stringsAsFactors = TRUE)[ , c(15, 1:14)]
colnames(adult) <- c("Y", paste0("X", 1:14))
adult$Y <- binary(adult$Y)

park <- "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
park <- read.csv(park)[ , c(18, 2:17, 19:24)]
colnames(park) <- c("Y", paste0("X", 1:22))
park$Y <- binary(park$Y)

higgs <- "https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv"
higgs <- read.csv(higgs)
colnames(higgs) <- c("Y", paste0("X", 1:28))
higgs$Y <- binary(higgs$Y)

bcds <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
bcds <- read.csv(bcds, stringsAsFactors = TRUE)
colnames(bcds) <- c("Y", paste0("X", 1:9))
bcds$Y <- binary(bcds$Y)

vote <- "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
vote <- read.csv(vote, stringsAsFactors = TRUE)
colnames(vote) <- c("Y", paste0("X", 1:16))
vote$Y <- binary(vote$Y)

clim <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
clim <- read.csv(clim, stringsAsFactors = TRUE, sep = "")[ , c(21, 3:20)]
colnames(clim) <- c("Y", paste0("X", 1:18))
clim$Y <- binary(clim$Y)

utils <- list( # data sets
  "phon" = phon,            # phonemes data set
  "spam" = spam,            # spam data set
  "wdbc" = wdbc,            # WI diagnostic breast cancer data set
  "adult" = adult,          # US Census Bureau data set
  "park" = park,            # Parkinson's data set
  "higgs" = higgs,          # Higgs boson data set (first 10K obs)
  "bcds" = bcds,            # breast cancer data set
  "vote" = vote,            # Congressional votes data set
  "clim" = clim,            # climate data set
  # helper functions
  "begin_process" = begin_process,    # process timing
  "end_process" = end_process,        # process timing
  "cv_parallel" = cv_parallel,        # parallel cross-validation wrapper
  # loss functions
  "Kappa" = Kappa,                    # Cohen's Kappa
  "MCC" = MCC,                        # Matthews correlation coefficient
  # data pre-processing
  "binary" = binary,                  # enforce binary factor encoding [0, 1]
  "ohe" = ohe,                        # one-hot encode, matrix output, no Y
  "ohe.data.frame" = ohe.data.frame,  # one-hot encode, df output, Y included
  "standardize" = standardize)        # standardize numeric variables

path <- "~/Documents/Github/Empirical_Sudy_of_Ensemble_Learning_Methods/utils/"

sapply(1:length(utils), function(x) {
  
  saveRDS(utils[[x]], file = paste0(path, names(utils)[[x]]))
  return(paste("saved", names(utils)[[x]]))
  
})

# check that utils have been saved

rm(list = ls())

path <- "~/Documents/Github/Empirical_Sudy_of_Ensemble_Learning_Methods/utils/"

files <- c("phon", "spam", "wdbc", "adult", "park", "higgs", "bcds", "vote", "clim",
           "begin_process", "end_process", "cv_parallel", "Kappa", "MCC",
           "binary", "ohe", "ohe.data.frame", "standardize")

for (i in 1:length(files)) { assign(files[i], readRDS(paste0(path, files[i]))) }
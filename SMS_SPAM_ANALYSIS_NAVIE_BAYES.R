#install.packages("quanteda") #popular package for statistical analysis of text data
require(quanteda)
require(RColorBrewer)
library(ggplot2)
library(tm)
library(class)
library(caret)
library(wordcloud)
library(e1071)
library(gmodels)
library(dplyr)
library(scales)
library(caret)
sms_raw1<-read.csv("C:\\Users\\hjall\\Desktop\\Data Science Materials\\Second_Semester 2023\\Machine Learning and Data Mining\\Assignment_Docs\\SMS DATA.csv", encoding = "latin1", stringsAsFactors=FALSE)

#attach(spam)
View(sms_raw1)
#View(sms_raw2)

# Reduce the dataset to only two columns (e.g., v1 and v2)
sms_raw <- sms_raw1[, c("v1", "v2")]
View(sms_raw)

# Initial Dataset Exploration
str(sms_raw)

# Convert categorical variable into factor
sms_raw$v1 <- as.factor(sms_raw$v1)

# EDA
# Descriptive Statistics
str(sms_raw)

# Create a frequency table
freq_table <- table(sms_raw$v1)
freq_table

# Create a bar graph
barplot(freq_table, col = c("red", "green"), main = "Frequency of Spam and Ham Messages", xlab = "Frequency", ylab = "Label")

# Calculate the proportions of spam and ham messages
proportions <- prop.table(table(sms_raw$v1))
# Convert the table object to a data frame
proportions_df <- as.data.frame(proportions)

# Use the fortify function on the data frame
proportions_df <- fortify(proportions_df, region = "Var1")

# Create a pie chart
ggplot(proportions_df, aes(x = "", y = Freq, fill = Var1)) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Proportion of Spam and Ham Messages", x = NULL, y = NULL, fill = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = paste0(Freq * 100, "%")), position = position_stack(vjust = 0.5))

# Creating a word cloud to visualize the text data
wordcloud(sms_raw, min.freq = 50, random.order = F)

# Creating a corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$v2))

# For potential additional options offered by the "tm" package:
#vignette("tm")

inspect(sms_corpus[1:5])

# to check for a single message we can use the as.character() function as well as double-brackets
as.character(sms_corpus[[400]])

# Using lapply() to print several messages
lapply(sms_corpus[1:10], as.character)

# converting the corpus to lower-case to start cleaning up the corpus
sms_corpus_clean <- tm_map(sms_corpus,
                           content_transformer(tolower))

# Comparing first sms to check result
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

# Removing numbers to reduce noise (numbers will be unique and will not provide useful patters across all messages)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# Check results
lapply(sms_corpus_clean[1:10], as.character)

# Removing "stop words" (to, and, but, or) and punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# Stemming
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# Remove additional whitespace
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# Tokenization
sms_dtm <- DocumentTermMatrix(sms_corpus_clean) # Thanks to the previous preprocessing the object is ready

# Data preparation - Train and Test ####
sms_dtm_train <- sms_dtm[1:4182,]
sms_dtm_test <- sms_dtm[4183:5572,]

sms_train_labels <- sms_raw[1:4182,]$v1
sms_test_labels <- sms_raw[4183:5572,]$v1

# Comparing proportion of SPAM
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# Creating a word cloud to visualize the text data
wordcloud(sms_corpus_clean, min.freq = 50, random.order = F)

# Creating subset of SPAM sms to visualize later
spam <- subset(sms_raw, v1 == "spam")
ham <- subset(sms_raw, v1 == "ham")

# Visualizing both types separately
wordcloud(spam$v2, max.words = 40, scale = c(3, 0.5), random.order = F)
wordcloud(ham$v2, max.words = 40, scale = c(3, 0.5), random.order = F)

# Data preparation - creating indicator features for frequent words####
# Filtering out unfrequent words
sms_freq_words <- findFreqTerms(sms_dtm_train, 5) # function to find all terms appearing at least 5 times
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq_words]

# Changing cells in sparse matrix to indicate yes/no since Naive Bayes typically works with Categorical features
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)

# Training model on the data ####
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)

CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = F, prop.t = F,
           dnn = c('predicted', 'actual'))

# Improving Model Performance

# Rebuilding Naive Bayes with laplace = 1
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels,
                              laplace = 1)

# Evaluating 2nd model's performance
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = F, prop.t = F, prop.r = F,
           dnn = c('predicted', 'actual'))

# Confusion matrix
conf_matrix <- confusionMatrix(data = sms_test_pred2, reference = sms_test_labels)
conf_matrix

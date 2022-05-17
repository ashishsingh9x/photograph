Practical No: 1 – K means clustering. Aim: Read a data file grades_km_input.csv and apply k-means clustering.
Data sample:
 

Practical 1 script:
# install required packages 
install.packages("plyr") 
install.packages("ggplot2") 
install.packages("cluster") 
install.packages("lattice") 
install.packages("grid") 
install.packages("gridExtra") 
# Load the package 
library(plyr) 
library(ggplot2) 
library(cluster) 
library(lattice) 
library(grid) 
library(gridExtra)
# A data frame is a two-dimensional array-like structure in which each column contains values of one
variable and each row contains one set of values from each column.
grade_input=as.data.frame(read.csv("D:/MSCIT-II/BIG DATA/Prac1.csv"))
kmdata_orig=as.matrix(grade_input[, c ("Student","English","Math","Science")])
kmdata=kmdata_orig[,2:4]
kmdata[1:10,]
# the k-means algorithm is used to identify clusters for k = 1, 2, .. . , 15. For each value of k, the WSS
is calculated.
wss=numeric(15)
# the option n start=25 specifies that the k-means algorithm will be repeated 25 times, each starting
with k random initial centroids
for(k in 1:15)wss[k]=sum(kmeans(kmdata,centers=k,nstart=25)$withinss)
plot(1:15,wss,type="b",xlab="Number of Clusters",ylab="Within sum of square")
#As can be seen, the WSS is greatly reduced when k increases from one to two. Another substantial
reduction in WSS occurs at k = 3. However, the improvement in WSS is fairly linear fork > 3.
km = kmeans(kmdata,3,nstart=25)
km
c( wss[3] , sum(km$withinss))
df=as.data.frame(kmdata_orig[,2:4])
df$cluster=factor(km$cluster)
centers=as.data.frame(km$centers)
g1=ggplot(data=df, aes(x=English, y=Math, color=cluster )) +
 geom_point() + theme(legend.position="right") +
 geom_point(data=centers,aes(x=English,y=Math, color=as.factor(c(1,2,3))),size=10, alpha=.3,
show.legend =FALSE)
g2=ggplot(data=df, aes(x=English, y=Science, color=cluster )) +
 geom_point () +geom_point(data=centers,aes(x=English,y=Science,
color=as.factor(c(1,2,3))),size=10, alpha=.3, show.legend=FALSE)
g3 = ggplot(data=df, aes(x=Math, y=Science, color=cluster )) +
 geom_point () + geom_point(data=centers,aes(x=Math,y=Science,
color=as.factor(c(1,2,3))),size=10, alpha=.3, show.legend=FALSE)
tmp=ggplot_gtable(ggplot_build(g1))
grid.arrange(arrangeGrob(g1 + theme(legend.position="none"),g2 +
theme(legend.position="none"),g3 + theme(legend.position="none"),top ="High School Student
Cluster Analysis" ,ncol=1))

ScreenShot:
 
 











Practical no 2: Apriori algorithm Aim: Perform Apriori algorithm using Groceries dataset from the R arules package.
Practical 2 script:
install.packages("arules")
install.packages("arulesViz")
install.packages("RColorBrewer")
# Loading Libraries
library(arules)
library(arulesViz)
library(RColorBrewer)
# import dataset
data(Groceries)
Groceries
summary(Groceries)
class(Groceries)
# using apriori() function
rules = apriori(Groceries, parameter = list(supp = 0.02, conf = 0.2))
summary (rules)
# using inspect() function
inspect(rules[1:10])
# using itemFrequencyPlot() function
arules::itemFrequencyPlot(Groceries, topN = 20,
 col = brewer.pal(8, 'Pastel2'),
 main = 'Relative Item Frequency Plot',
 type = "relative",
 ylab = "Item Frequency (Relative)")
itemsets = apriori(Groceries, parameter = list(minlen=2, maxlen=2,support=0.02, target="frequent
itemsets"))
summary(itemsets)
# using inspect() function
inspect(itemsets[1:10])
itemsets_3 = apriori(Groceries,parameter = list(minlen=3, maxlen=3,support=0.02, target="frequent
itemsets"))
summary(itemsets_3)
# using inspect() function
inspect(itemsets_3)

Output:
 
With itemset min 2
 
With min support 3
 
Practical no: 3 Linear regression
Practical no: 3 a) Simple Linear regression 
Aim: Create your own data for years of experience and salary in lakhs and apply linear regression model to predict the salary
Practical 3 script:
years_of_exp = c(7,5,1,3)
salary_in_lakhs = c(16,10,6,8)

employee.data = data.frame(years_of_exp, salary_in_lakhs)
employee.data
model <- lm(salary_in_lakhs ~ years_of_exp, data = employee.data)
summary(model)
plot(salary_in_lakhs ~ years_of_exp, data = employee.data)
abline(model)
 
 
















b) Logistic regression 
Aim: Take the in-built data from ISLR package and apply generalized logistic regression to find whether a person would be defaulter or not; considering input as student, income and balance.
Practical 3 b script:

#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
print (sample)
train <- data[sample, ]
test <- data[!sample, ]
nrow(train)
nrow(test)
# Fit the Logistic Regression Model
# use the glm (general linear model) function and specify family="binomial"
#so that R fits a logistic regression model to the dataset
model <- glm(default~student+balance+income, family="binomial", data=train)
#view model summary
summary(model)
#Model Diagnostics
install.packages("InformationValue")
library(InformationValue)
predicted <- predict(model, test, type="response")
confusionMatrix(test$default, predicted)

 

















Practical 4 a Decision Tree
Data Structure:
 
Practical 4 script
dataset = read.csv('D:/MSCIT-II/BIG DATA/pract4.csv')
dataset = dataset[3:5]
# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))
# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])
# Fitting Decision Tree Classification to the Training set
install.packages('rpart')
library(rpart)
classifier = rpart(formula = Purchased ~ .,
 data = training_set)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')
# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)
# Visualising the Training set results
install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
 main = 'Decision Tree Classification (Training set)',
 xlab = 'Age', ylab = 'Estimated Salary',
 xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3], main = 'Decision Tree Classification (Test set)',
 xlab = 'Age', ylab = 'Estimated Salary',
 xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
plot(classifier)
text(classifier)
Output:
 

 










Practical no: 4b Naïve Bayes Classification:
Practical 4b script:
# Naive Bayes
# Importing the dataset
dataset = read.csv('D:/MSCIT-II/BIG DATA/pract4.csv')
dataset = dataset[3:5]
# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))
# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])
# Fitting Naive Bayes to the Training set
install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-3],
 y = training_set$Purchased)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])
# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)
print(cm)
# Visualising the Training set results
install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
print(set)
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
 main = 'Naive Bayes (Training set)',xlab = 'Age', ylab = 'Estimated Salary',
 xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'NaiveBayes (Test set)',
 xlab = 'Age', ylab = 'Estimated Salary',
 xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

output:
 

 










Practical 5: Text Analysis
Practical 5 Script:
dataset_original = read.delim("D:/MSCIT-II/BIG DATA/Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)
# Cleaning the texts
install.packages('tm')
install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked
print(dataset$Liked)
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# Fitting Random Forest Classification to the Training set
install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
 y = training_set$Liked,
 ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

cm = table(test_set[, 692], y_pred)
print(cm)
Output:
 






Practical 6. Aim: Install configure and run Hadoop and HDFS.
Step to install Hadoop:

•	Install Java JDK 1.8
•	Download Hadoop and extract and place under C drive
•	Set Path in Environment Variables
•	Config files under Hadoop directory
•	Create folder datanode and namenode under data directory
•	Edit HDFS and YARN files
•	Set Java Home environment in Hadoop environment
•	Setup Complete. Test by executing start-all.cmd
In this practical Hadoop will be configure for single node.
1.	Download Hadoop
https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.3.0/hadoop-3.3.0.tar.gz

extract to C:\Hadoop
2.	Set the path JAVA_HOME Environment variable
3.	Set the path HADOOP_HOME Environment variable








 

 
 
 
 
 


















 



5.	Configurations
Edit file C:/Hadoop-3.3.0/etc/hadoop/core-site.xml,
paste the xml code in folder and save
 
<configuration>
   <property>
       <name>fs.defaultFS</name>
       <value>hdfs://localhost:9000</value>
   </property>
</configuration>
======================================================
 
 “mapred-site.xml” and edit this file C:/Hadoop-3.3.0/etc/hadoop/mapred-site.xml, paste xml code and save this file.
 
<configuration>
   <property>
       <name>mapreduce.framework.name</name>
       <value>yarn</value>
   </property>
</configuration>
======================================================
 
Create folder “data” under “C:\Hadoop-3.3.0”
Create folder “datanode” under “C:\Hadoop-3.3.0\data”
Create folder “namenode” under “C:\Hadoop-3.3.0\data”
 
======================================================
Edit file C:\Hadoop-3.3.0/etc/hadoop/hdfs-site.xml,
paste xml code and save this file.
 
<configuration>
   <property>
       <name>dfs.replication</name>
       <value>1</value>
   </property>
   <property>
       <name>dfs.namenode.name.dir</name>
       <value>/hadoop-3.3.0/data/namenode</value>
   </property>
   <property>
       <name>dfs.datanode.data.dir</name>
       <value>/hadoop-3.3.0/data/datanode</value>
   </property>
</configuration>
======================================================
 
Edit file C:/Hadoop-3.3.0/etc/hadoop/yarn-site.xml,
paste xml code and save this file.
 
<configuration>
   <property>
                <name>yarn.nodemanager.aux-services</name>
                <value>mapreduce_shuffle</value>
   </property>
   <property>
               <name>yarn.nodemanager.auxservices.mapreduce.shuffle.class</name> 
                <value>org.apache.hadoop.mapred.ShuffleHandler</value>
   </property>
</configuration>
======================================================
 
Edit file C:/Hadoop-3.3.0/etc/hadoop/hadoop-env.cmd
by closing the command line
“JAVA_HOME=%JAVA_HOME%” instead of set “JAVA_HOME=C:\Java”
 
======================================================
6.	Hadoop Configurations
Download
https://github.com/s911415/apache-hadoop-3.1.0-winutils
– Copy folder bin and replace existing bin folder in
C:\Hadoop-3.3.0\bin
– Format the NameNode
– Open cmd and type command “hdfs namenode –format”
 
7.	Testing
– Open cmd and change directory to C:\Hadoop-3.3.0\sbin
– type start-all.cmd
 
– Start namenode and datanode with this command
– type start-dfs.cmd
– Start yarn through this command
– type start-yarn.cmd
 
Open: http://localhost:8088
 
Open: http://localhost:9870
 











Practical 7: installing Mongo and manipulating it with query.
Steps for installing mongo:
Download mongo db from: https://www.mongodb.com/try#community
Step 1: click on the downloaded exe.
Step 2: below screen will load.
 
Follow the below step:
 
 


 
 
 
After the whole installation is done, the user must configure it.
Step 1: Go to the local disk C and get into “Program Files“. There you’ll find a folder named “MongoDB“.
 
Step 2: Open it and you’ll find a folder named “bin” i.e. binaries folder. You will have 15 to 17 files in it. Copy the path, as given in the snippet path i.e. C:\Program Files\MongoDB\Server\4.0\bin
 
Step 3: Open Settings and search “Path” please refer practical 7.

Step 7: Open Command prompts and type “mongod” to start the service.
 
Step 8: Write command on the command prompt “mongo” to create the connection.
 

Basic Query:
1.	Query for help.: MongoDB Help
 
2.	 Show All Databases: show dbs
 
3.	Create new database: use DATABASE_NAME
 
4.	 Know your current selected database: db
 
5.	Drop database: db.dropDatabase()
 

6.	Insert document in collection>db.COLLECTION_NAME.insert(document)
 
7.	Get collection document >db.COLLECTION_NAME.find()
 
8.	Update document >db.COLLECTION_NAME.update(SELECTION_CRITERIA, UPDATED_DATA)

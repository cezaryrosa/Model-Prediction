install.packages("plyr")
install.packages("corrplot")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("ggthemes")
install.packages("caret")
install.packages("MASS")
install.packages("party")
install.packages("tidyverse")
install.packages("Hmisc")
install.packages("ggplot2")
install.packages("magrittr")
install.packages("dplyr")
install.packages('patchwork')
install.packages('tidyverse')
install.packages("vcd")
install.packages("caret")


library(plyr)
library(corrplot) #nie ma 
library(ggplot2)
library(gridExtra) #nie ma
library(ggthemes) #nie ma 
library(caret)
library(MASS)
library(party)
library (tidyverse)
library (Hmisc)
library (ggplot2)
library(magrittr)
library(dplyr)
library(patchwork)
library(tidyverse)
library(vcd)
library(caret)

getwd()
churn <- read.csv("/Users/cezaryrosa/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv", header = TRUE, sep=",")
str(churn)

#missing values
sapply(churn, function(x) sum(is.na(x)))
#remove 11 missing records
churn <- churn[complete.cases(churn), ]
#remove costumerID since we won't use it
churn$customerID <- NULL
# continuous variables statistics - tenure, MonthlyCharges, TotalCharges
describe(churn) 


# Plot - target var: Churn
churn %>% 
  group_by(Churn) %>% 
  summarise(Number = n()) %>%
  mutate(Percent = prop.table(Number)*100) %>% 
  ggplot(aes(Churn, Percent)) + 
  geom_col(aes(fill = Churn)) +
  labs(title = "Churn Percentage") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(label = sprintf("%.2f%%", Percent)), hjust = 0.01,vjust = -0.5, size = 4) +
  theme_minimal()

####################################2. Data Exploration (EDA)####################################

#2.1 Numerical Variables

plot1 <- ggplot(churn, aes(churn$MonthlyCharges, fill = Churn)) + geom_histogram()
plot(plot1)
plot2 <- ggplot(churn, aes(churn$TotalCharges, fill = Churn)) + geom_histogram()
plot(plot2)
plot3 <- ggplot(churn, aes(churn$tenure, fill = Churn)) + geom_histogram()
plot(plot3)

p1 <- ggplot(churn, aes(x=gender)) + ggtitle("Gender") + xlab("Gender") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()

#Categorical Variables
cat_var <- churn %>% select(-tenure, -MonthlyCharges, -TotalCharges)
cat <- data.frame(cat_var)

head(cat)

theme_set(theme_void())

pieplotter <- function(col) {
  tibble(var = col) %>% 
    count(var) %>% 
    mutate(
      p = n/sum(n),
      y_mid = lag(cumsum(p), default = 0) + (p/2) 
    ) %>% 
    ggplot() +
    geom_col(
      aes(x = "", y = p, fill = var)
    ) +
    coord_polar(theta = "y") +
    geom_text(
      aes(x = "", y = y_mid, label = scales::percent(p))
    ) +
    theme(
      axis.text.x = element_blank()
    )
}

map(cat, pieplotter)


#2.2 MultiPlot - target var: Churn

num_var <- churn %>% select(tenure, MonthlyCharges, TotalCharges)

install.packages("cowplot")
library(cowplot)

options(repr.plot.width = 12, repr.plot.height = 8)
plot_grid(ggplot(churn, aes(x=gender,fill=Churn))+ geom_bar()+ theme_bw(),
          ggplot(churn, aes(x=SeniorCitizen,fill=Churn))+ geom_bar(position = 'fill')+theme_bw(),
          ggplot(churn, aes(x=Partner,fill=Churn))+ geom_bar(position = 'fill')+theme_bw(),
          ggplot(churn, aes(x=Dependents,fill=Churn))+ geom_bar(position = 'fill')+theme_bw(),
          ggplot(churn, aes(x=PhoneService,fill=Churn))+ geom_bar(position = 'fill')+theme_bw(),
          ggplot(churn, aes(x=MultipleLines,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          align = "h")

options(repr.plot.width = 12, repr.plot.height = 8)
plot_grid(ggplot(churn, aes(x=InternetService,fill=Churn))+ geom_bar(position = 'fill')+ theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          ggplot(churn, aes(x=OnlineSecurity,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          ggplot(churn, aes(x=OnlineBackup,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          ggplot(churn, aes(x=DeviceProtection,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          ggplot(churn, aes(x=TechSupport,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          ggplot(churn, aes(x=StreamingTV,fill=Churn))+ geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          align = "h")

#################################### 3. Correlation Check####################################

#3.1 Checking correlation between variables, drawing correlation matrix using Spearman's Rank Correlation.
#3.2 Spearman correlation matrix for continuous variables: Total Charges, Tenure, Montly Charges

mydata.cor <- cor(num_var, use = "complete.obs", method = "spearman")
install.packages("corrplot")
library(corrplot)
corrmap <- corrplot(mydata.cor, method = "color", addCoef.col = 'black')

##As for continuous variables, let's take a closer look at them.
pairs(num_var, pch = 18, col = "red") #tego nie uzylam ostatecznie
ggpairs(num_var)

#People having lower tenure and higher monthly charges tend to churn more.

#3.3. Cramers V correlation matrix for categorical variables

cat_var %>% drop_na()

catcor <- function(x, type=c("cramer", "phi", "contingency")) {
  require(vcd)
  nc <- ncol(x)
  v <- expand.grid(1:nc, 1:nc)
  type <- match.arg(type)
  res <- matrix(mapply(function(i1, i2) assocstats(table(x[,i1],
                                                         x[,i2]))[[type]], v[,1], v[,2]), nc, nc)
  rownames(res) <- colnames(res) <- colnames(x)
  res
}

catcor(cat_var, type="phi")

catcordf <- as.data.frame(catcor)

#Findings:
  
#1) Total Charges and Tenure are highly correlated (0.89).
# Considering that TotalCharges can be derived from tenure*Monthly Charges, we can just drop the Total Charges.

#2) Streaming TV, Streaming Movies services are quite correlated between each other (0.53)
# Considering the EDA results, the distribution of values (1/0 and also churn distribution) of Streaming TV and Movies were almost same. So, we decided to assemble those two and make it into one variable 'Streaming Services'.

#3) PhoneService and Multiple lines are very correlated between each other (correlation = 1)
# We will drop PhoneService (Bcz it's less correlated with our Target Variable Churn than the Multiple lines).

#4) Contract is 41% correlated with Churn. We should take it into consideration and investigate in later steps.


# Drop 'Total Charges' and 'PhoneService' based on the result of Correlation matrix
churn_new <- churn %>% select(-TotalCharges, -PhoneService)

#Let's check if there is any relationship between tenure and MonthlyCharges (so between numerical variables)

lm(churn_new$MonthlyCharges ~ churn_new$tenure, data=subset(churn_new, Churn=0))

ggplot(churn_new, aes(x = churn_new$tenure, y = churn_new$MonthlyCharges)) + 
  geom_point()


###### 4.Data Partitioning (Split)######

#Cleaning the Categorical features

churn <- data.frame(lapply(churn, function(x) {
  gsub("No internet service", "No", x)}))

churn <- data.frame(lapply(churn, function(x) {
  gsub("No phone service", "No", x)}))

#Creating Numerical Variables and normalizing features

num_var <- churn[,c("tenure", "MonthlyCharges", "TotalCharges")]
num_var$tenure = as.numeric(as.factor(num_var$tenure))
num_var$MonthlyCharges = as.numeric(as.factor(num_var$MonthlyCharges))
num_var$TotalCharges = as.numeric(as.factor(num_var$TotalCharges))
num_var <- data.frame(scale(num_var))

#Creating Dummy Variables
churn_dummy<- data.frame(sapply(cat_var,function(x) data.frame(model.matrix(~x-1,data =cat_var))[,-1]))

head(churn_dummy)

#Combining the data
churn_final <- cbind(num_var,churn_dummy)
head(churn_final)

set.seed(123)
install.packages("caTools")
library(caTools)

indices = sample.split(churn_final$Churn, SplitRatio = 0.7)
train = churn_final[indices,]
test = churn_final[!(indices),]

#Building the logistic model using all variables to select significant variables
model_1 = glm(Churn ~ ., data = train, family = "binomial")
summary(model_1)

#Building the final model using significant variables

model_2 = glm(Churn ~ tenure + MonthlyCharges + TotalCharges +
              InternetService.xFiber.optic + InternetService.xNo +
                OnlineSecurity.xYes +
                OnlineBackup.xYes +
              TechSupport.xYes +
              Contract.xOne.year + Contract.xTwo.year + PaperlessBilling +
              PaymentMethod.xElectronic.check, data = train, family = "binomial")
summary(model_2)

final_model <- model_2

#Model Evaluation using test set

pred <- predict(final_model, type = "response", newdata = test[,-12])
summary(pred)
test$prob <- pred

pred_churn <- factor(ifelse(pred >= 0.50, "Yes", "No"))
actual_churn <- factor(ifelse(test$Churn==1,"Yes","No"))
table(actual_churn,pred_churn)


train_selected <- train %>% select(tenure, MonthlyCharges, TotalCharges,
                                     InternetService.xFiber.optic, InternetService.xNo,
                                     OnlineSecurity.xYes,
                                     OnlineBackup.xYes,
                                     TechSupport.xYes,
                                     Contract.xOne.year, Contract.xTwo.year, PaperlessBilling,
                                     PaymentMethod.xElectronic.check, Churn)

head(train_selected)
describe(train_selected)

test_selected <- train %>% select(tenure, MonthlyCharges, TotalCharges,
                                   InternetService.xFiber.optic, InternetService.xNo,
                                   OnlineSecurity.xYes,
                                   OnlineBackup.xYes,
                                   TechSupport.xYes,
                                   Contract.xOne.year, Contract.xTwo.year, PaperlessBilling,
                                   PaymentMethod.xElectronic.check, Churn)
head(test_selected)
describe(test_selected)

#Remerging train and test sets
rejoined_churn_final <- merge(train_selected, test_selected, by = c('tenure', 'MonthlyCharges', 'TotalCharges',
                                                                    'InternetService.xFiber.optic', 'InternetService.xNo', 'OnlineSecurity.xYes',
                                                                    'OnlineBackup.xYes',
                                                                    'TechSupport.xYes',
                                                                    'Contract.xOne.year', 'Contract.xTwo.year', 'PaperlessBilling',
                                                                    'PaymentMethod.xElectronic.check', 'Churn'))

head(rejoined_churn_final)

nrow(rejoined_churn_final)
ncol(rejoined_churn_final)

#Variable Clustering - K-means clustering method

library(purrr)

#Elbow Method - 1st method

# Use map_dbl to run many models with varying value of k (centers)
tot_withinss <- map_dbl(1:10,  function(k){
  model <- kmeans(x = rejoined_churn_final, centers = k)
  model$tot.withinss
})

# Generate a data frame containing both k and tot_withinss
elbow_df <- data.frame(
  k = 1:10,
  tot_withinss = tot_withinss
)

# Plot the elbow plot
ggplot(elbow_df, aes(x = k, y = tot_withinss)) +
  geom_line() +
  scale_x_continuous(breaks = 1:10)


#Elbow Method - 2nd method

set.seed(123)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(rejoined_churn_final, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")



#Average Silhouette Method - 1st method

install.packages("cluster")
library(cluster)

# function to compute average silhouette for k clusters
avg_sil <- function(k) {
  km.res <- kmeans(rejoined_churn_final, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(rejoined_churn_final))
  mean(ss[, 3])
}

# Compute and plot wss for k = 2 to k = 15
k.values <- 2:15

# extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")

#Average Silhouette Method - 2nd method

install.packages("factoextra")
library(factoextra)

fviz_nbclust(rejoined_churn_final, kmeans, method = "silhouette")

#Silhouette plot of pam

library(cluster)

# Generate a k-means model using the pam() function with a k = 2
pam_k2 <- pam(rejoined_churn_final, k = 2)

# Plot the silhouette visual for the pam_k2 model
plot(silhouette(pam_k2))

# Generate a k-means model using the pam() function with a k = 4
pam_k3 <- pam(rejoined_churn_final, k = 3)

# Plot the silhouette visual for the pam_k4 model
plot(silhouette(pam_k3))

#Silhouette analysis allows you to calculate how similar each observations is with the cluster it is assigned relative to other clusters. This metric (silhouette width) ranges from -1 to 1 for each observation in your data and can be interpreted as follows:
#Values close to 1 suggest that the observation is well matched to the assigned cluster
#Values close to 0 suggest that the observation is borderline matched between two clusters
#Values close to -1 suggest that the observations may be assigned to the wrong cluster

#Gap Statistic Method

# compute gap statistic
set.seed(123)
gap_stat <- clusGap(rejoined_churn_final, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
# Print the result
print(gap_stat, method = "firstmax")

fviz_gap_stat(gap_stat)

set.seed(120)
n_clusters <-2
churn_cluster <- kmeans(na.omit(rejoined_churn_final[,c(1,3)]), centers=n_clusters, nstart = 20)

#Print caharacteristics of cluster
churn_cluster$centers
churn_cluster$size

#Create dataset to be plot, with five clusters
cluster <- c(1: n_clusters)
center_df <- data.frame(cluster, churn_cluster$centers)
# Reshape the data
center_reshape <- gather(center_df, tenure,values, tenure:TotalCharges)

#Create the palette
library(RColorBrewer)
hm.palette <-colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')

#plot data
ggplot(data = center_reshape, aes(x = tenure, y = cluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 5, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = hm.palette(90)) +
  theme_classic() +
  labs( x ="Features")

#Based on Monthly Charges and Tenure, there is five types of clusters.

# Compute k-means model with k = 4
set.seed(123)
model_km <- kmeans(rejoined_churn_final, centers = 4, nstart = 25)

# Print the results
print(model_km$centers)

#From the results we can see that:
  
#961 observations were assigned to the first cluster
#1572 observations were assigned to the second cluster
#1154 observations were assigned to the third cluster
#1387 observations were assigned to the fourth cluster

#plot results of final k-means model
fviz_cluster(model_km, data = rejoined_churn_final[,c(1,2,3)])

#We can visualize the clusters on a scatterplot that displays the first two principal components on the axes using the fivz_cluster() function:


dist_churn <- dist(rejoined_churn_final)
dist_churn
hc_churn <- hclust(dist_churn)
hc_churn
clust_churn <- cutree(hc_churn, h = 1)
clust_churn
segment_churn <- mutate(rejoined_churn_final, cluster = clust_churn)

# Count the number of observations that fall into each cluster
count(segment_churn, cluster)

# Color the dendrogram based on the height cutoff
library(dendextend)
dend_churn <- as.dendrogram(hc_churn)
dend_colored <- color_branches(dend_churn, h = 2)

# Plot the colored dendrogram
plot(dend_colored)

# Calculate the mean for each category
segment_churn %>% 
  group_by(cluster) %>% 
  summarise_all(list(mean))


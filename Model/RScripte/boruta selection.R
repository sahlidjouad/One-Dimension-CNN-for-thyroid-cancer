
dataset <- read.csv("C:\\Users\\User\\Documents\\Python programm\\streamlit\\StanfordAnalysis\\dataset\\column_selected_by_lasso_mrmr.csv" )

dataset <- subset(dataset, select=-X)

str(dataset)

library(Boruta)
set.seed(111)


Boruta(Target~., data = dataset, doTrace=2, maxRuns=25000) -> boruta.dataset


print(boruta.dataset)


boruta.data<- TentativeRoughFix(boruta.dataset)
print(boruta.data)

bank_df <- attStats(boruta.data)
print(bank_df)
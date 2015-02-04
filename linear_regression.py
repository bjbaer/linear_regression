import numpy as np
import pandas as pd
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

g = lambda x: float(x.replace("%", "")) #create lambda function to remove % and convert to float
loansData['Interest.Rate'] = map(g, loansData['Interest.Rate']) #overwrite old interest rate

h = lambda x: int(x.replace(" months","")) #create lambda function to replace ' months' and turn into integer
loansData['Loan.Length'] = map(h, loansData['Loan.Length'])

i = lambda x : int((x.split('-'))[1]) #Create a lambda function to split the range, take the first value, and convert to integer
loansData['FICO.Range'] = map(i, loansData['FICO.Range'])

intrate = loansData['Interest.Rate'] #create new variable based on cleaned interest rate
loanamt = loansData['Amount.Requested'] #create new variable based on cleaned loan Amount
fico = loansData['FICO.Range'] #create a new variable based on clean fico amount
#transpose the variables into columns
y = np.matrix(intrate).transpose() 
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1, x2]) #create input matrix
#create the linear model
X = sm.add_constant(x) #Big x at the start
model = sm.OLS(y,X) #Ordinary Least Squares Regression
f = model.fit()

print 'Coefficients: ', f.params[1:3]
print "Intercept: ", f.params[0]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared
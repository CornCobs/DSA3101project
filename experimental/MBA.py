import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from itertools import permutations
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("correct_data.csv")

df = pd.DataFrame(data)

df = (df.groupby(["Panel ID", "Date"]))

list_of_unique = list(df["Category"].unique())

flattened = [i for t in list_of_unique  for i in t]
groceries = list(set(flattened))
rules = list(permutations(groceries, 2))
rules_df = pd.DataFrame(rules, columns=['antecedents', 'consequents'])

print(rules)

encoder = TransactionEncoder().fit(list_of_unique)

onehot = encoder.transform(list_of_unique)

onehot = pd.DataFrame(onehot, columns = encoder.columns_)

support = onehot.mean()

print(onehot.head())

print(support)

def support(x):
	# Compute support for antecedent AND consequent
	support = x.mean()

	return support

# Define an empty list for Support
supportA_metric = []
supportC_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Support and append it to the list
	supportA_metric.append(support(antecedent))
	supportC_metric.append(support(consequent))
    
# Print results
rules_df['antecedent support'] = supportA_metric
rules_df['consequent support'] = supportC_metric


def support_2(antecedent, consequent):
	# Compute support for antecedent AND consequent
	supportAC = np.logical_and(antecedent, consequent).mean()

	return supportAC

# Define an empty list for Support
support_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Support and append it to the list
	support_metric.append(support_2(antecedent, consequent))
    
# Print results
rules_df['support'] = support_metric


def confidence(antecedent, consequent):
	# Compute confidence for antecedent AND consequent
	confidence = support_2(antecedent, consequent)/support(antecedent)

	return confidence

# Define an empty list for confidence
confidence_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Confidence and append it to the list
	confidence_metric.append(confidence(antecedent, consequent))
    
# Print results
rules_df['confidence'] = confidence_metric

def lift(antecedent, consequent):
	# Compute lift for antecedent AND consequent
	lift = support_2(antecedent, consequent)/(support(antecedent) * support(consequent))

	return lift

# Define an empty list for Support
lift_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Lift and append it to the list
	lift_metric.append(lift(antecedent, consequent))
    
# Print results
rules_df['lift'] = lift_metric

def leverage(antecedent, consequent):
	# Compute leverage for antecedent AND consequent
	leverage = support_2(antecedent, consequent) - (support(antecedent) * support(consequent))

	return leverage

# Define an empty list for Support
leverage_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Leverage and append it to the list
	leverage_metric.append(leverage(antecedent, consequent))
    
# Print results
rules_df['leverage'] = leverage_metric


def conviction(antecedent, consequent):
	# Compute support for antecedent AND consequent
	supportAC = np.logical_and(antecedent, consequent).mean()

	# Compute support for antecedent
	supportA = antecedent.mean()

	# Compute support for NOT consequent
	supportnC = 1.0 - consequent.mean()

	# Compute support for antecedent and NOT consequent
	supportAnC = supportA - supportAC

    # Return conviction
	return supportA * supportnC / supportAnC


# Define an empty list for Conviction
conviction_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Conviction and append it to the list
	conviction_metric.append(conviction(antecedent, consequent))
    
# Print results
rules_df['conviction'] = conviction_metric


# Define a function to compute Zhang's metric
def zhang(antecedent, consequent):
	# Compute the support of each book
	supportA = antecedent.mean()
	supportC = consequent.mean()

	# Compute the support of both books
	supportAC = np.logical_and(antecedent, consequent).mean()

	# Complete the expressions for the numerator and denominator
	numerator = supportAC - supportA*supportC
	denominator = max(supportAC*(1-supportA), supportA*(supportC-supportAC))

	# Return Zhang's metric
	return numerator / denominator

# Define an empty list for Zhang's metric
zhangs_metric = []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
	antecedent = onehot[itemset[0]]
	consequent = onehot[itemset[1]]
    
    # Complete Zhang's metric and append it to the list
	zhangs_metric.append(zhang(antecedent, consequent))
    
# Print results
rules_df['zhang'] = zhangs_metric

#rules_df.to_csv (r'C:\Users\User\Desktop\NUS\Year 3 Sem 1\DSA3101\Hackerthon\test.csv', index = False, header=True)

sns.scatterplot(x = 'support', y = 'confidence', data = rules_df)

plt.show()

# Set the lift threshold to 1.5
rules_df = rules_df[rules_df['lift'] > 1.5]

# Set the conviction threshold to 1.0
rules_df = rules_df[rules_df['conviction'] > 1.0]

# Set the threshold for Zhang's rule to 0.65
rules_df = rules_df[rules_df['zhang'] > 0.65]

sns.scatterplot(x="support", y="confidence", data = rules_df)

plt.show()

print(rules_df)



'''
coords = rules_to_coordinates(rules_df)

# Generate parallel coordinates plot
parallel_coordinates(coords, 'rule')
plt.legend([])
plt.show()

'''


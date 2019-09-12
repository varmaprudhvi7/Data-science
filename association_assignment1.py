import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
#----------------EDA pipeline--------------------------
#removing the unwanted columns and creating dummy variables
movies.drop(['V1','V2','V3','V4','V5'],axis=1,inplace=True)
from collections import Counter
item_frequencies = Counter(movies)
#---------------MODEL BUILDING--------------------------
#with support 0.005
frequent_itemsets = apriori(movies, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
import matplotlib.pyplot as plt
plt.bar(list(range(1,10)),frequent_itemsets.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_itemsets.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')
#rules for confidence
rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules1.head(20)
rules1.sort_values('confidence',ascending = False,inplace=True)
#rules for support
rules11 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules11.head(20)
rules11.sort_values('lift',ascending = False,inplace=True)
#----------------MODEL 2--------------------------
#with support 0.01
frequent_itemsets1 = apriori(movies, min_support=0.01, max_len=2,use_colnames = True)
frequent_itemsets1.sort_values('support',ascending = False,inplace=True)
plt.bar(list(range(1,10)),frequent_itemsets1.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_itemsets1.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')
#rules based on lift ratio
rules2 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules2.head(20)
rules2.sort_values('lift',ascending = False,inplace=True)
#rules based on confidence
rules22 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules22.head(20)
rules22.sort_values('confidence',ascending = False,inplace=True)
#----------------------redundancy----------------
def to_list(i):
    return (sorted(list(i)))


ma_X = rules11.antecedents.apply(to_list)+rules11.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
# getting rules without any redudancy 
rules_no_redudancy  = rules11.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)

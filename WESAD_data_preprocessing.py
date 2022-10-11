def quantile_normalize(df):
    df_sorted = pd.DataFrame(np.sort(df.values,axis = 0),index = df.index,columns = df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1,len(df_mean)+1)
    df_qn = df.rank(method = "min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)


df_0 = baseline_0.tail(10000)
#baseline
# df_1_l = baseline.tail(10000)
df_1 = baseline.tail(10000)
# df_1 = df_1.drop(columns = 'w_labels',axis = 1)
# df_1 = quantile_normalize(df_1)
# display(df_1)

#stress 
# df_2_l = stress.tail(10000)
df_2 = stress.tail(10000)
# df_2 = df_2.drop(columns = 'w_labels',axis = 1)
# df_2 = quantile_normalize(df_2)
# display(df_2)

#amusement 
# df_3_l = amusement.tail(10000)
df_3 = amusement.tail(10000)
# df_3 = df_3.drop(columns = 'w_labels',axis = 1)
# df_3 = quantile_normalize(df_3)
# display(df_3)

df_chest = df_3.append(df_1.append(df_2.append(df_0,ignore_index = True),ignore_index = True),ignore_index = True)
labels_wesad = df_chest['w_labels'].to_numpy()
df_chest = df_chest.drop(columns = 'w_labels',axis = 1)
df_chest = quantile_normalize(df_chest)
display(df_chest)


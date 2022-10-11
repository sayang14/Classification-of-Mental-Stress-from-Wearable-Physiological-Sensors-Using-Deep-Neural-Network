def quantile_normalize(df):
    df_sorted = pd.DataFrame(np.sort(df.values,axis = 0),index = df.index,columns = df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1,len(df_mean)+1)
    df_qn = df.rank(method = "min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)

x_train_swell = quantile_normalize(X_train)
x_test_swell = quantile_normalize(X_test)

display(x_train_swell)


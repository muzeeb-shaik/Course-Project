import pandas as pd
import researchpy as rp
############### McNemar Test
#os.chdir('D:\\Muzeeb\\ML Class\\Project\\CSCE633_S20_Ananya_Gargi_Keerat_Muzeeb_Project')

ffnn = pd.read_csv("predicted_values_ffnn.txt")
cnn = pd.read_csv("predicted_values_cnn.txt")
ffnn.columns = ['ffnn']
cnn.columns = ['cnn']
df = pd.DataFrame()
df = pd.concat([cnn,ffnn], axis = 1)
rp.summary_cat(df[['ffnn', 'cnn']])


table, res = rp.crosstab(df['ffnn'], df['cnn'], test= 'mcnemar')
print("******************************************************")
print("McNemar's Test CNN vs FFNN for 2 Class Classification")

print(table)

print(res)
print("******************************************************")

### 3 Class

ffnn_3 = pd.read_csv("predicted_values_ffnn_3class.txt")
cnn_3 = pd.read_csv("predicted_values_cnn_3class.txt")
ffnn_3.columns = ['ffnn']
cnn_3.columns = ['cnn']
df_3 = pd.DataFrame()
df_3 = pd.concat([ffnn_3,cnn_3], axis = 1)
rp.summary_cat(df_3[['ffnn', 'cnn']])


table_3, res_3 = rp.crosstab(df_3['ffnn'], df_3['cnn'], test= 'mcnemar')

print("******************************************************")
print("McNemar's Test CNN vs FFNN for 3 Class Classification")

print(table_3)

print(res_3)
print("******************************************************")
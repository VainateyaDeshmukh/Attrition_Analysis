import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py
# Importing data for analysis
df = pd.read_csv("D:\\BA_College\\3rd Sem Assignment\\Attrition.csv")
#Bar_Plot
sns.barplot(x="Gender",y="Attrition",data=df)

#Pie Chart
trace = go.Pie(labels=['No_attrition', 'Yes_attrition'], values=df['Attrition'].value_counts(),
               textfont=dict(size=15), opacity=0.8,
               marker=dict(colors=['lightskyblue', 'gold'],
                           line=dict(color='#000000', width=1.5)))

layout = dict(title='Distribution of attrition variable')

fig = dict(data=[trace], layout=layout)
py.plot(fig)

#Heat Map
sns.set(font_scale=1.5)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.xticks(rotation=90)
plt.show()
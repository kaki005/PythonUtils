import pandas as pd
from utilpy import draw_cd_diagram

df_perf = pd.read_csv("data/example.csv", index_col=False)
draw_cd_diagram(df_perf=df_perf, title="Accuracy", labels=True)

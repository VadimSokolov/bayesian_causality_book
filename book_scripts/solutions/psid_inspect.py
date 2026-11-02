import pandas as pd
d = pd.read_stata('data/psid_controls.dta')
print(d.shape, d.columns.tolist()); print(d.describe().T[['mean','min','max']].round(2)); print(d.head(3))

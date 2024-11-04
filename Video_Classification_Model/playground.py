
import os, pathlib, re
import pandas as pd

# one
file = "Sports3999.csv"
file_df= pd.concat([pd.read_csv(file,header=None).assign(New=os.path.basename(file).split('.')[0]) 
        ])
file_df.columns = ['name', 'time', 'size','label']  

file_df['label'] = file_df['label'].apply(lambda x: re.sub(r'^([a-zA-Z]+).*', r'\1', x))
file_df.to_csv(os.path.join("./out-old.csv"))  # set the output file location and name at the sametime


# two

file_path = pathlib.Path(file)
df = pd.read_csv(file_path, header=None, names=['name', 'time', 'size'])
df['label'] = file_path.stem
df['label'] = df['label'].str.extract(r'^([a-zA-Z]+)', expand=False)
df.to_csv(os.path.join("./out-new.csv"))
# print(df)


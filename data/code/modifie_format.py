import pandas as pd

df = pd.read_csv("data/datasets/train_v2_drcat_02.csv")
df = df[['text', 'label']]

df_1 = df[df['label'] == 1].copy()
df_0 = df[df['label'] == 0].copy()

df_1['source'] = 'machine'
df_0['source'] = 'human'

print(df.shape)
print(df.shape)



df_0.to_json("data/datasets/drcat_human.jsonl", orient="records", lines=True)
df_1.to_json("data/datasets/drcat_machine.jsonl", orient="records", lines=True)

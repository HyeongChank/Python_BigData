import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()

titanic.to_csv('./data/titanic.csv', index=False)

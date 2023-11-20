from sklearn import preprocessing


AAMI = ['N', 'L', 'R', 'V', 'A', '|', 'B']
le_aami = preprocessing.LabelEncoder()
le_aami.fit(AAMI)
AAMI_encoded = le_aami.transform(AAMI)
decoded = list(le_aami.inverse_transform([2, 2, 1]))

print(decoded)
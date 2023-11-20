from pymongo import MongoClient
import csv

def connection():
    conn = MongoClient("localhost")
    db = conn.ECGdb
    set1 = db.ECG_2
    # set1.delete_many({})
    return set1

def insertToMongoDB(set1):
    with open('Y_EU_ST-T_V5_2.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        array_y = []
        for label in reader:
            for i in label:
                array_y.append(label[i])

    with open('X_EU_ST-T_V5_2.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        counts = 0
        for signal in reader:
            array_x = []
            for k in signal:
                array_x.append(float(signal[k]))
            data = {
                'fs': 250,
                'from': 'european-st-t-database-1.0.0',
                'lead': 'V5',
                'p_signal': array_x,
                'label': array_y[counts],
                'resample': True
            }
            print(counts)
            print(array_x)
            print(array_y[counts])
            counts += 1
            set1.insert_one(data)
            print('成功添加了' + str(counts) + '条数据 ')

def main():
    set1 = connection()
    insertToMongoDB(set1)

if __name__ == '__main__':
    main()

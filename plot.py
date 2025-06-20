import pandas as pd
import matplotlib.pyplot as plt
import sys

def rolling_average_plot(csv_file, window_size, rows=-1):
    data = pd.read_csv(csv_file, header=None, names=['values'])
    if rows != -1:
        data = data.iloc[:rows]
    data['rolling_avg'] = data['values'].rolling(window=window_size).mean()
    plt.plot(data['rolling_avg'])
    return len(data)

if __name__ == "__main__":
    csv_file = sys.argv[1]
    window_size = int(sys.argv[2])
    title = sys.argv[3]

    plt.figure(figsize=(10, 5))
    rows = rolling_average_plot(csv_file, window_size)
    plt.xlabel('Numer bitwy')
    plt.ylabel('Nagroda')
    plt.title(title)
    plt.show()
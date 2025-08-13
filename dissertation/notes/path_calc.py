import pandas as pd
import math
from scipy.spatial.distance import cdist

def parse_path(values):
    def get_order(values):
        indexed = [(val, idx) for idx, val in enumerate(values)]
        indexed.sort()  # 按 val 排序
        order = [idx for val, idx in indexed]
        return order

    order1 = get_order(values)

    order1.append(0)
    return order1

# parse_path([0.1, 1.0, 0.9, 0.2, 0.4, 0.5, 0.8, 0.7, 0.3, 0.6])

def calculate_path_distance(visits, coordinates):
    if not visits or len(visits) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(len(visits) - 1):
        x1, y1 = coordinates[visits[i]]
        x2, y2 = coordinates[visits[i + 1]]
        total_distance += math.hypot(x2 - x1, y2 - y1)
    return total_distance

def cal_distance(visits, coordinates):
    matrix = cdist(coordinates, coordinates)
    total_dist = 0.0
    for i in range(len(visits) - 1):
        total_dist += matrix[visits[i], visits[i + 1]]
    return total_dist

import json
def add_distances_to_dataframe(df):
    def compute_distances(row):
        try:
            state = json.loads(row['state'])
            coords = state['coordinates']
            order_embedding = state['order_embedding']
            visits = parse_path(order_embedding)
            d1 = calculate_path_distance(visits, coords)
            d2 = cal_distance(visits, coords)
            return pd.Series({'distance1': d1, 'distance2': d2, 'visits': visits})
        except Exception as e:
            print(f"Error processing row: {e}")
            return pd.Series({'distance1': None, 'distance2': None, 'visits': None})

    df[['distance1', 'distance2','visits']] = df.apply(compute_distances, axis=1)
    return df


import pandas as pd

data = {
    'state': [
        {
            "current_city": 0.0,
            "coordinates": [
                [0.8231103407097919, 0.026117981569867332],
                [0.21077063993129397, 0.6184217693496102],
                [0.09828446533689916, 0.6201313098768588],
                [0.053890219598443756, 0.9606540578042385],
                [0.9804293742150735, 0.5211276502712239],
                [0.6365533448355478, 0.7647569482692499],
                [0.7649552946168192, 0.41768557955972274],
                [0.7688053063237427, 0.4232017504120317],
                [0.9261035715268315, 0.6819264848723984],
                [0.3684555913246884, 0.85890985535282]
            ],
            "current_city_onehot": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "visited_mask": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "order_embedding": [0.1, 1.0, 0.9, 0.8, 0.4, 0.6, 0.2, 0.3, 0.5, 0.7],
            "distances_from_current": [0.0, 0.7038438168314193, 0.7742397138193915, 1.0, 0.42912189404971013, 0.6294091421780603, 0.3270522676202873, 0.3311148579876853, 0.5484543412534502, 0.7838898600884837]
        }
    ]
}

df = pd.read_csv('01.csv')
df = add_distances_to_dataframe(df)
# df = add_distances_to_dataframe(df[(df['episode']==1)&(df['done']==1)&(df['state_type']=='full')])
df.to_csv('002.csv', index=False)
print(df[['distance1', 'distance2']])

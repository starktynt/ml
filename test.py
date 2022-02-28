import pandas as pd
data={
        "user_query":[],
        "query_type":[],
        "data_match":[],
        "similar":[],
        "similar1":[],
        "similar2":[],
        "time_taken":[],
        "score":[]
    }
df = pd.DataFrame(data)
df.to_csv("record.csv")


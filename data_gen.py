import pandas as pd
import numpy as np
data = {
    'ID': np.arange(1, 21), 
    'Name': [f'Person_{i}' for i in range(1, 21)], 
    'Age': np.random.randint(18, 65, 20),  
    'Score': np.random.uniform(0, 100, 20),  
    'Category': np.random.choice(['A', 'B', 'C'], 20) 
}
df = pd.DataFrame(data)
print(df)

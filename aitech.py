import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data (replace this with your own dataset)
# For example, you might have data on student performance and teaching methods
# Here we generate random data for demonstration purposes
np.random.seed(0)
num_students = 1000
teaching_methods = ['Traditional', 'AI Enhanced']
teaching_data = {
    'Teaching Method': np.random.choice(teaching_methods, num_students),
    'Study Hours': np.random.randint(1, 10, num_students),
    'Exam Score': np.random.randint(50, 100, num_students)
}
df = pd.DataFrame(teaching_data)

# Encode teaching method as numerical values
df['Teaching Method'] = df['Teaching Method'].map({'Traditional': 0, 'AI Enhanced': 1})

# Split data into features (X) and target variable (y)
X = df[['Teaching Method', 'Study Hours']]
y = df['Exam Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Analyze the effect of AI on teaching
coef_ai = model.coef_[0]  # Coefficient for AI Enhanced teaching method
print("Effect of AI on teaching (Coefficient):", coef_ai)

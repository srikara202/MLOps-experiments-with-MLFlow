import mlflow
print("Printing tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://localhost:5000")
print("Printing new tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")
import torch
from torch import nn

class torch_LogisticRegression(nn.Module):
  """
  Logistic Regression model using PyTorch nn.Module
  """
  def __init__(self):
    super(torch_LogisticRegression, self).__init__()
    #self.linear = nn.Linear(input_size, output_size)

  def forward(self, x):
    """
    Performs the linear transformation followed by sigmoid activation
    """
    y_pred = torch.sigmoid(self.linear(x))
    return y_pred

  def fit(self, X, y):
      
    try:
        input_dim=X.shape[1]
    except IndexError:
        input_dim=X.unsqueeze(1).shape[1]
        
    try:    
        output_dim=y.shape[1]
    except IndexError:
        output_dim=y.unsqueeze(1).shape[1]

    self.linear =  nn.Sequential(
    nn.Linear(input_dim, input_dim),
    nn.ReLU(),
    nn.Linear(input_dim, int(0.6*input_dim)),
    nn.ReLU(),
    nn.Linear(int(0.6*input_dim),output_dim),
    )
    #nn.Linear(input_dim,output_dim)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for logistic regression
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

    # Train the model
    epochs = 1000
    for epoch in range(epochs):
      # Forward pass
      y_pred = self.forward(X)
      loss = criterion(y_pred, y.float().unsqueeze(1))  # Convert labels to float for loss calculation

      # Backward pass and update weights
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print training information (optional)
      if (epoch+1) % 100 == 0:
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}')  
      

# Example usage
# Define some data (replace with your actual data)
X = torch.randn(100, 5)  # 100 samples, 5 features
y = torch.randint(0, 2, (100,))  # Binary labels (0 or 1)

# Create the model
model = torch_LogisticRegression()  # Input size 5, output size 1 (binary)
model.fit(X,y)


# Use the trained model for prediction
new_data = torch.randn(1, 5)  # Example data for prediction
predicted_proba = model(new_data).data.item()  # Get probability from sigmoid output

print(f'Predicted probability for new data: {predicted_proba:.4f}')

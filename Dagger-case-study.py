import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# --- Expert function ---
def expert_function(x):
    return torch.sin(x) + 0.5 * x**3 - 2 * x**2 + 3

# --- Student model ---
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Add Gaussian noise ---
def add_noise(x, std=0.5):
    return x + torch.randn_like(x) * std

# --- Train student model ---
def train_student(model, x_noisy, y_target, epochs=2000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_noisy)
        loss = loss_fn(pred, y_target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# --- DAGGER imitation loop ---
def dagger_train(student, x_seed, expert_fn, noise_std=0.5, dagger_iters=50):
    x_clean = x_seed.clone().detach()
    y_expert = expert_fn(x_clean)
    x_noisy = add_noise(x_clean, std=noise_std)

    dataset_x = [x_noisy]
    dataset_y = [y_expert]

    print("Initial training on seed data:")
    train_student(student, x_noisy, y_expert)

    for i in range(dagger_iters):
        print(f"\n--- DAGGER Iteration {i+1}/{dagger_iters} ---")

        # Student proposes new noisy inputs
        x_proposed = add_noise(x_clean, std=noise_std)

        # Expert gives labels for clean inputs
        y_expert_new = expert_fn(x_clean)

        print(f"  Sample input x_proposed[:5]:\n    {x_proposed[:5].squeeze()}")
        print(f"  Expert output y_expert_new[:5]:\n    {y_expert_new[:5].squeeze()}")

        # Aggregate data
        dataset_x.append(x_proposed)
        dataset_y.append(y_expert_new)

        # Combine all data
        all_x = torch.cat(dataset_x, dim=0)
        all_y = torch.cat(dataset_y, dim=0)

        print(f"  Dataset size: {len(all_x)} samples")

        # Retrain student
        train_student(student, all_x, all_y)

    print("\nTraining complete.")
    return student
# --- Main execution ---
def test(student,expert_function):
    
        student.eval()
        for i in range(10):
            # Sample random integer and convert to tensor
            x = torch.tensor([[np.random.randint(-5, 5)]], dtype=torch.float32)  # Shape (1,1)

            # Get predictions
            with torch.no_grad():
                student_pred = student(x)
                expert_val = expert_function(x)

            print(f"Input: {x.item():>5} | Student: {student_pred.item():>8.4f} | Expert: {expert_val.item():>8.4f}")
    
        
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a seed dataset
    x_seed = torch.arange(-5.0, 6.0, 1.0).unsqueeze(1)  # [-5, ..., 5]
    student = StudentNet()
    trained_student = dagger_train(student, x_seed, expert_function)
    test(trained_student, expert_function)
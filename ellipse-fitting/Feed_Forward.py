import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import warnings

# --- PyTorch Imports for Neural Network ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.random.seed(42)
torch.manual_seed(42)

# --- CONSTANTS ---
MIN_HITS = 10         # Minimum photons required to process an event
IMAGE_BINS = 64       
MAX_COORD = 350       # Assumed detector range in mm, for binning
IMG_RANGE = [[-MAX_COORD, MAX_COORD], [-MAX_COORD, MAX_COORD]]
MAX_ANGLE = 45.0      # Used to normalize target angles for stable training

# --- GAUSSIAN SMEARING PARAMETERS ---
# Standard deviation for Gaussian smearing in mm
POSITION_SMEAR_SIGMA = 0.0  # mm 

# --- NEURAL NETWORK DEFINITION ---

class AngleRegressor(nn.Module):
    """A simple Feed-Forward Neural Network to regress the angle."""
    def __init__(self, input_size):
        super(AngleRegressor, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Output is a single value: the angle
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

class CherenkovDataset(Dataset):
    """Custom PyTorch Dataset for our Cherenkov hit 'images'."""
    def __init__(self, images, angles):
        self.images = torch.FloatTensor(images)
        self.angles = torch.FloatTensor(angles)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.angles[idx]


# --- DATA PROCESSING WITH NEURAL NETWORK ---

def apply_gaussian_smearing(x, y, sigma):
    if sigma <= 0:
        return x, y
    
    # Generate Gaussian noise
    noise_x = np.random.normal(0, sigma, size=x.shape)
    noise_y = np.random.normal(0, sigma, size=y.shape)
    
    # Add noise to original coordinates
    x_smeared = x + noise_x
    y_smeared = y + noise_y
    
    return x_smeared, y_smeared

def process_cherenkov_data_nn(filename):
    """
    Processes Cherenkov data using a neural network to reconstruct angles.
    Now includes Gaussian smearing of position data.
    """
    df = pd.read_csv(filename)
    if 'angle_deg' not in df.columns:
        print("Error: 'angle_deg' column not found, which is required for training.")
        return pd.DataFrame(), {}, {}

    grouped = df.groupby(['run_id', 'event_id'])

    stats = {
        'total_events': len(grouped),
        'events_processed': 0,
        'failed_min_hits': 0,
        'smearing_applied': POSITION_SMEAR_SIGMA > 0,
        'smear_sigma': POSITION_SMEAR_SIGMA
    }

    # 1. Pre-process data: Convert each event's hits into a 2D histogram (image)
    print("Step 1: Pre-processing data into fixed-size images...")
    if POSITION_SMEAR_SIGMA > 0:
        print(f"  - Applying Gaussian smearing with σ = {POSITION_SMEAR_SIGMA} mm")
    else:
        print("  - No position smearing applied")
    
    all_images = []
    all_angles = []
    all_info = [] # To keep track of run/event ids and photon counts

    for (run_id, event_id), group in grouped:
        x_original, y_original = group['x_mm'].values, group['y_mm'].values
        
        if len(x_original) < MIN_HITS:
            stats['failed_min_hits'] += 1
            continue
            
        stats['events_processed'] += 1
        true_angle = group['angle_deg'].iloc[0]

        # Apply Gaussian smearing to the coordinates
        x_smeared, y_smeared = apply_gaussian_smearing(x_original, y_original, POSITION_SMEAR_SIGMA)

        # Create a 2D histogram from the smeared hit coordinates
        hist, _, _ = np.histogram2d(x_smeared, y_smeared, bins=IMAGE_BINS, range=IMG_RANGE)
        
        # Normalize the image - this helps the NN focus on the shape
        if hist.sum() > 0:
            hist /= hist.sum()

        all_images.append(hist)
        all_angles.append(true_angle)
        all_info.append({
            'run_id': run_id,
            'event_id': event_id,
            'true_angle': true_angle,
            'num_photons': len(x_original),
            'raw_hits': (x_smeared, y_smeared),  # Store smeared hits for plotting
            'original_hits': (x_original, y_original)  # Keep original for comparison if needed
        })

    if not all_images:
        print("No events met the minimum hit requirement.")
        return pd.DataFrame(), {}, stats

    all_images = np.array(all_images)
    # Normalize angles to be in [0, 1] for better training stability
    all_angles = np.array(all_angles) / MAX_ANGLE

    # 2. Split data into training and validation sets
    print(f"Step 2: Splitting {len(all_images)} events into training and validation sets...")
    indices = np.arange(len(all_images))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_val = all_images[train_indices], all_images[val_indices]
    y_train, y_val = all_angles[train_indices], all_angles[val_indices]

    train_dataset = CherenkovDataset(X_train, y_train.reshape(-1, 1))
    val_dataset = CherenkovDataset(X_val, y_val.reshape(-1, 1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 3. Initialize and train the neural network
    print("Step 3: Initializing and training the neural network...")
    input_size = IMAGE_BINS * IMAGE_BINS
    model = AngleRegressor(input_size)
    criterion = nn.MSELoss() # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 250 
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, angles in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                outputs = model(images)
                loss = criterion(outputs, angles)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), 'cherenkov_model.pth')
    
    # Plot training history (Additional Relevant Information)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Model Training History (Position Smearing σ = {POSITION_SMEAR_SIGMA} mm)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Use the trained model to predict angles for all events
    print("Step 4: Generating predictions for all events...")
    model.eval()
    with torch.no_grad():
        all_images_tensor = torch.FloatTensor(all_images)
        predictions_normalized = model(all_images_tensor).numpy().flatten()

    # De-normalize the predictions back to degrees
    reconstructed_angles = predictions_normalized * MAX_ANGLE

    # 5. Assemble the final results DataFrame
    results_list = []
    for i, info in enumerate(all_info):
        info['reconstructed_angle'] = reconstructed_angles[i]
        results_list.append(info)
    
    results_df = pd.DataFrame(results_list)

    # Prepare examples for the grid plot (similar logic to original)
    examples = {}
    results_df_copy = results_df.copy()
    results_df_copy['angle_bin'] = results_df_copy['true_angle'].round()
    
    for bin_key, group in results_df_copy.groupby('angle_bin'):
        if bin_key not in examples:
            first_entry = group.iloc[0]
            examples[bin_key] = (
                first_entry['raw_hits'][0],    # x-coords (smeared)
                first_entry['raw_hits'][1],    # y-coords (smeared)
                first_entry['true_angle'],
                first_entry['reconstructed_angle']
            )

    return results_df.drop(columns=['raw_hits', 'original_hits']), examples, stats


if __name__ == "__main__":
    file_pattern = "../cherenkov_hits_*.csv"
    try:
        files = glob.glob(file_pattern)
        if not files:
            raise FileNotFoundError(f"No file found matching the pattern '{file_pattern}'.")
        filename = files[0]
        print(f"Found data file: '{filename}'\n")

        results_df, examples, stats = process_cherenkov_data_nn(filename)
        
        print_summary_statistics_nn(results_df, stats)

        if not results_df.empty:
            print(f"Now generating plots...")
            
            plot_examples_grid(examples)
            plot_results(results_df)
        else:
            print("\nANALYSIS FAILED: No events could be processed by the NN.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your data CSV file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
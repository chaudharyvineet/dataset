import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import socket
import os
from datetime import datetime, timedelta

# Define the NetworkTrafficDataset
class NetworkTrafficDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading data from {csv_file}...")
        self.original_data = pd.read_csv(csv_file)
        self.data = self.original_data.copy()
        print(f"Data loaded with shape {self.data.shape}")

        # Store protocol distribution for later use
        if 'protocol' in self.data.columns:
            protocol_counts = self.data['protocol'].value_counts()
            self.protocol_probabilities = protocol_counts / len(self.data)
            print("Protocol distribution in original data:")
            for protocol, prob in self.protocol_probabilities.items():
                print(f"Protocol {protocol}: {prob:.3f}")

        # Convert timestamp column to datetime
        if 'timestamp' in self.original_data.columns:
            self.original_data['timestamp'] = pd.to_datetime(self.original_data['timestamp'])
            self.start_time = self.original_data['timestamp'].min()
            self.time_range = (self.original_data['timestamp'].max() - self.start_time).total_seconds()
            print(f"Original data time range: {self.time_range / 86400:.2f} days")

        # Convert IP addresses to integers
        print("Converting IP addresses...")
        self.data['src_ip'] = self.data['src_ip'].apply(self.ip_to_int)
        self.data['dst_ip'] = self.data['dst_ip'].apply(self.ip_to_int)

        # Drop non-numeric columns and convert the remaining to numeric
        self.non_numeric_columns = ['direction', 'timestamp', 'Label']
        columns_to_drop = [col for col in self.non_numeric_columns if col in self.data.columns]
        self.data = self.data.drop(columns=columns_to_drop)

        print("Converting data to numeric format...")
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)

        # Store normalization parameters and normalize data to [0, 1]
        self.feature_mins = self.data.min()
        self.feature_maxs = self.data.max()
        range_diff = self.feature_maxs - self.feature_mins
        range_diff[range_diff == 0] = 1  # Prevent division by zero
        self.data = (self.data - self.feature_mins) / range_diff

        self.features = self.data.values.astype(np.float32)

        # Process labels if they exist
        if 'Label' in self.original_data.columns:
            self.labels = self.original_data['Label'].apply(lambda x: 1 if str(x).lower() == 'malicious' else 0).values.astype(np.float32)
        else:
            self.labels = np.zeros(len(self.features), dtype=np.float32)

        print(f"Preprocessing complete. Features shape: {self.features.shape}")

    def ip_to_int(self, ip):
        try:
            if pd.isna(ip) or not isinstance(ip, str):
                return 0
            return struct.unpack("!I", socket.inet_aton(ip.strip()))[0]
        except:
            return 0

    def int_to_ip(self, ip_int):
        try:
            return socket.inet_ntoa(struct.pack("!I", int(ip_int)))
        except:
            return "0.0.0.0"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

    def format_generated_data(self, generated_data):
        df = pd.DataFrame(generated_data, columns=self.data.columns)
        df = df * (self.feature_maxs - self.feature_mins) + self.feature_mins

        # Define protocol mapping
        protocol_mapping = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}

        # Assign protocol values if 'protocol' column exists
        if 'protocol' in df.columns:
            # Retrieve protocol probabilities with defaults to 0 if missing
            protocol_probs = [
                self.protocol_probabilities.get(6, 0),
                self.protocol_probabilities.get(17, 0),
                self.protocol_probabilities.get(1, 0)
            ]

            # Normalize probabilities to sum to 1
            total_prob = sum(protocol_probs)
            if total_prob > 0:
                protocol_probs = [p / total_prob for p in protocol_probs]
            else:
                protocol_probs = [1 / 3, 1 / 3, 1 / 3]  # Fallback to equal distribution if all probabilities are zero

            # Apply probabilities to assign protocol values
            df['protocol'] = df['protocol'].apply(lambda x: np.random.choice([6, 17, 1], p=protocol_probs))
            df['protocol'] = df['protocol'].apply(lambda x: protocol_mapping.get(x, 'Unknown'))

        # Convert specified columns to integer and round values
        numeric_columns = {
            'src_port': 'int',
            'dst_port': 'int',
            'flags': 'int',
            'packet_length': 'int',
            'packets': 'int',
            'bytes': 'int'
        }
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = df[col].round().astype(dtype)

        # Convert IP addresses from integer format to string format if columns are present
        if 'src_ip' in df.columns:
            df['src_ip'] = df['src_ip'].apply(self.int_to_ip)
        if 'dst_ip' in df.columns:
            df['dst_ip'] = df['dst_ip'].apply(self.int_to_ip)

        # Randomly assign directions based on unique values in the original data
        if 'direction' in self.original_data.columns:
            df['direction'] = np.random.choice(self.original_data['direction'].unique(), size=len(df))

        # Generate timestamps within a specified range
        if 'timestamp' in self.original_data.columns:
            start_time = datetime.now()
            time_range = timedelta(days=2.5)  # Total time range of 2.5 days
            timestamps = np.linspace(0, time_range.total_seconds(), num=len(df))
            df['timestamp'] = [start_time + timedelta(seconds=float(t)) for t in timestamps]
            df = df.sort_values('timestamp')  # Sort by timestamp for chronological order

        # Assign a default label of 'Normal' if the 'Label' column exists in original data
        if 'Label' in self.original_data.columns:
            df['Label'] = 'Normal'

        # Reindex to match the original data column order
        return df.reindex(columns=self.original_data.columns)


# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()  # Assuming output is normalized
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator model using Conv1d layers
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Flatten(),
            nn.Linear(256 * input_size, 1),  # Ensure input_size matches
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        return self.model(x)

# Training function for GAN
def train_gan(generator, discriminator, data_loader, epochs, optimizer_g, optimizer_d, device):
    criterion = nn.BCELoss()
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        total_d_loss, total_g_loss, num_batches = 0, 0, 0

        for batch_idx, (real_data, _) in enumerate(data_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Real and Fake Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            ## Train Discriminator ##
            discriminator.zero_grad()

            # Real Data
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            # Fake Data
            noise = torch.randn(batch_size, 100).to(device)  # Latent dimension is 100
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            d_loss_fake = criterion(fake_output, fake_labels)

            # Total Discriminator Loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            ## Train Generator ##
            generator.zero_grad()

            # Generate Fake Data
            noise = torch.randn(batch_size, 100).to(device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)

            # Generator tries to fool the discriminator
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

            # Accumulate Losses
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

        # Average Losses for the Epoch
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        print(f"Epoch [{epoch + 1}/{epochs}] Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}")

# Function to generate synthetic data using the trained generator
def generate_synthetic_data(generator, num_samples, device, dataset):
    """Generate synthetic network traffic data"""
    generator.eval()
    with torch.no_grad():
        batch_size = 1000
        num_batches = (num_samples + batch_size - 1) // batch_size
        generated_data = []

        print("\nGenerating synthetic data...")
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            noise = torch.randn(current_batch_size, 100).to(device)  # Latent dimension is 100
            fake_data = generator(noise)
            generated_data.append(fake_data.cpu().numpy())

            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Generated {(i + 1) * batch_size} / {num_samples} samples...")

        generated_data = np.vstack(generated_data)[:num_samples]  # Ensure exact number of samples
        print("Formatting generated data...")
        return dataset.format_generated_data(generated_data)

# Main function to execute the GAN workflow
def main():
    # Configuration
    CONFIG = {
        'input_file': 'one_day_data_A.csv',  # Your input CSV file
        'output_dir': 'generated_data',
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'latent_dim': 100,
        'generation_multiplier': 2.5  # Generate 2.5x the original dataset size
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Initialize dataset and data loader
    dataset = NetworkTrafficDataset(CONFIG['input_file'])
    data_loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2  # Adjust based on your system
    )

    # Initialize models
    input_size = dataset.features.shape[1]
    generator = Generator(input_size=CONFIG['latent_dim'], output_size=input_size)
    discriminator = Discriminator(input_size=input_size)

    # Setup optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.5, 0.999)
    )

    # Train the GAN
    train_gan(
        generator,
        discriminator,
        data_loader,
        CONFIG['epochs'],
        optimizer_g,
        optimizer_d,
        device
    )

    # Generate synthetic data
    num_samples = int(len(dataset) * CONFIG['generation_multiplier'])
    generated_df = generate_synthetic_data(generator, num_samples, device, dataset)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(CONFIG['output_dir'], f'synthetic_network_traffic_{timestamp}.csv')
    generated_df.to_csv(output_file, index=False)
    print(f"\nSynthetic data saved to {output_file}")

    # Save models
    torch.save(generator.state_dict(),
               os.path.join(CONFIG['output_dir'], f'generator_{timestamp}.pth'))
    torch.save(discriminator.state_dict(),
               os.path.join(CONFIG['output_dir'], f'discriminator_{timestamp}.pth'))
    print("Models saved successfully")

if __name__ == "__main__":
    main()

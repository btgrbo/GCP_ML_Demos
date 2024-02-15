import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data set into the environment
df = pd.read_csv(r"C:\Users\OliverNowakbtelligen\OneDrive - b.telligent group\Desktop\Tickets\GCP ML Demos\python\demo2\test\training.csv")

# Get summary statistics for numerical columns
summary_statistics = df.describe()
print(summary_statistics)

# Create boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Age'])
plt.title('Boxplot for Age')
plt.show()

# Creating histograms
plt.figure(figsize=(10, 4))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')  # Adjust bins and other parameters as needed
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# Function to loop through the data set for visual inspection
def create_boxplots_and_histograms(df):
    # Iterate through columns
    for column in df.columns:
        # Check if the column has integer type
        if df[column].dtype == 'int64':
            # Create boxplot
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot for {column}')

            # Create histogram
            plt.subplot(1, 2, 2)
            sns.histplot(df[column], bins=20, kde=True)
            plt.title(f'Histogram for {column}')

            # Adjust layout and show plots
            plt.tight_layout()
            plt.show()

# Example usage:
# Assuming df is your pandas df
# create_boxplots_and_histograms(df)

create_boxplots_and_histograms(df["Age"])


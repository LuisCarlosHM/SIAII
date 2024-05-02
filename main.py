import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt

def create_partitions(dataset, train_percentage, num_partitions, filename):
    # Create the partitions
    for i in range(num_partitions):
        # Shuffle the dataset
        shuffled_dataset = dataset.sample(frac=1, random_state=i)
        
        # Calculate the number of samples for training and test data
        num_samples = len(shuffled_dataset)
        num_train_samples = int(train_percentage * num_samples)
        num_test_samples = num_samples - num_train_samples
        
        # Split the dataset into training and test data
        train_data = shuffled_dataset[:num_train_samples]
        test_data = shuffled_dataset[num_train_samples:num_train_samples+num_test_samples]
        
        # Save the partitions to separate files without header
        train_data.to_csv(f'{filename}_train_{i+1}.csv', index=False, header=False)
        test_data.to_csv(f'{filename}_test_{i+1}.csv', index=False, header=False)

# Create a function to plot the dataset results of training and test data

def plot_dataset(dataset, train_data, test_data, filename):
    # Plot the training data
    plt.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], c=train_data.iloc[:, 2], cmap='coolwarm')
    plt.title(f'Training data for {filename}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    # Plot the test data
    plt.scatter(test_data.iloc[:, 0], test_data.iloc[:, 1], c=test_data.iloc[:, 2], cmap='coolwarm')
    plt.title(f'Test data for {filename}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    # Plot the dataset
    plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=dataset.iloc[:, 2], cmap='coolwarm')
    plt.title(f'Dataset for {filename}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def main():
    # Load the dataset
    dataset = pd.read_csv('spheres1d10.csv')

    # Set the percentage of training and test data
    train_percentage = 0.8

    # Set the number of partitions
    num_partitions = 5

    filename = 'spheres1d10'

    # Create the partitions
    create_partitions(dataset, train_percentage, num_partitions, filename)

    # Train and test the perceptron for each partition

    # Initialize the perceptron
    perceptron = Perceptron(num_inputs=3)

    # Train and test the perceptron for each partition
    
    for i in range(num_partitions):
        # Load the training and test data
        train_data = pd.read_csv(f'{filename}_train_{i+1}.csv')
        test_data = pd.read_csv(f'{filename}_test_{i+1}.csv')

        # Extract the inputs and labels
        training_inputs = train_data.iloc[:, :-1].values
        training_labels = train_data.iloc[:, -1].values
        test_inputs = test_data.iloc[:, :-1].values
        test_labels = test_data.iloc[:, -1].values


        # Train the perceptron
        perceptron.train(training_inputs, training_labels, num_epochs=100)

        # Test the perceptron
        accuracy = perceptron.test(test_inputs, test_labels)
        print(f'Accuracy for partition {i+1}: {accuracy}')  
        plot_dataset(dataset, train_data, test_data, filename)

    # Load the perturbed dataset
    perturbed_dataset_10 = pd.read_csv('spheres2d10.csv')
    perturbed_dataset_50 = pd.read_csv('spheres2d50.csv')
    perturbed_dataset_70 = pd.read_csv('spheres2d70.csv')

    # Change the label value to 1 if the inputs are -1, 1 and -1

    # perturbed_dataset_10.loc[(perturbed_dataset_10.iloc[:, 0] == -1) & (perturbed_dataset_10.iloc[:, 1] == 1) & (perturbed_dataset_10.iloc[:, 2] == -1), 3] = 1

    # Set the percentage of training and test data
    train_percentage = 0.8

    # Set the number of partitions
    num_partitions = 10

    # Create the filename variable for each file

    filename10 = 'spheres2d10'
    filename50 = 'spheres2d50'
    filename70 = 'spheres2d70'

    # Create the partitions for each perturbed dataset
    create_partitions(perturbed_dataset_10, train_percentage, num_partitions, filename10)
    create_partitions(perturbed_dataset_50, train_percentage, num_partitions,  filename50)
    create_partitions(perturbed_dataset_70, train_percentage, num_partitions, filename70)


    # Initialize the perceptron
    perceptron = Perceptron(num_inputs=3)

    # Train and test the perceptron for each partition of each perturbed dataset
    for i in range(num_partitions):
        # Load the training and test data for each perturbed dataset
        train_data_10 = pd.read_csv(f'{filename10}_train_{i+1}.csv')
        test_data_10 = pd.read_csv(f'{filename10}_test_{i+1}.csv')
        train_data_50 = pd.read_csv(f'{filename50}_train_{i+1}.csv')
        test_data_50 = pd.read_csv(f'{filename50}_test_{i+1}.csv')
        train_data_70 = pd.read_csv(f'{filename70}_train_{i+1}.csv')
        test_data_70 = pd.read_csv(f'{filename70}_test_{i+1}.csv')
        
        # Extract the inputs and labels for each perturbed dataset
        training_inputs_10 = train_data_10.iloc[:, :-1].values
        training_labels_10 = train_data_10.iloc[:, -1].values
        test_inputs_10 = test_data_10.iloc[:, :-1].values
        test_labels_10 = test_data_10.iloc[:, -1].values
        
        training_inputs_50 = train_data_50.iloc[:, :-1].values
        training_labels_50 = train_data_50.iloc[:, -1].values
        test_inputs_50 = test_data_50.iloc[:, :-1].values
        test_labels_50 = test_data_50.iloc[:, -1].values
        
        training_inputs_70 = train_data_70.iloc[:, :-1].values
        training_labels_70 = train_data_70.iloc[:, -1].values
        test_inputs_70 = test_data_70.iloc[:, :-1].values
        test_labels_70 = test_data_70.iloc[:, -1].values
        
        # Train the perceptron for each perturbed dataset
        perceptron.train(training_inputs_10, training_labels_10, num_epochs=100)
        perceptron.train(training_inputs_50, training_labels_50, num_epochs=100)
        perceptron.train(training_inputs_70, training_labels_70, num_epochs=100)
        
        # Test the perceptron for each perturbed dataset
        accuracy_10 = perceptron.test(test_inputs_10, test_labels_10)
        accuracy_50 = perceptron.test(test_inputs_50, test_labels_50)
        accuracy_70 = perceptron.test(test_inputs_70, test_labels_70)
        
        print(f'Accuracy for partition {i+1} with 10% perturbed dataset: {accuracy_10}')
        print(f'Accuracy for partition {i+1} with 50% perturbed dataset: {accuracy_50}')
        print(f'Accuracy for partition {i+1} with 70% perturbed dataset: {accuracy_70}')


  
if __name__ == "__main__":
    main()
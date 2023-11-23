import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Converter:
    def __init__(self, _rows):
        self.rows = _rows

    def convert(self, _input):
        """
        Shifts each letter in given text by one position to the left by QWERTY keyboard keys locations
        :param _input: Text
        :return: Shifted text
        """
        output = ""
        for letter in _input:
            if letter == ' ':
                output += letter
                continue

            index = self.find_index(letter)
            output += self.rows[index[0]][index[1] - 1]

        return output

    def find_index(self, letter):
        """
        Finds letter indexes in 2d array
        :param letter: Letter
        :return: Found indexes
        """
        for i, rows in enumerate(self.rows):
            for j, item in enumerate(rows):
                if item == letter:
                    return [i, j]

def visualize_shifted_text(shifted_text):
    # Create a bar plot using Seaborn to visualize the frequency of each letter in the shifted text
    plt.figure(figsize=(10, 6))
    sns.countplot(list(shifted_text), palette="viridis")
    plt.title("Frequency of Letters in Shifted Text")
    plt.xlabel("Letters")
    plt.ylabel("Frequency")
    plt.show()

def create_neural_network(input_size, output_size):
    # Create a simple neural network using TensorFlow/Keras
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train, y_train, epochs=10):
    # Train the neural network
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

def main():
    # QWERTY keyboard rows
    qwerty_rows = [
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm"
    ]

    # Example usage of the Converter class
    converter = Converter(qwerty_rows)
    original_text = "hello world"
    shifted_text = converter.convert(original_text)
    print(f"Original Text: {original_text}")
    print(f"Shifted Text: {shifted_text}")

    # Visualize the shifted text
    visualize_shifted_text(shifted_text)


    X_train = np.array([converter.find_index(letter) for letter in original_text if letter != ' '])
    y_train = np.array([converter.find_index(letter) for letter in shifted_text if letter != ' '])

    X_train_one_hot = tf.one_hot(X_train.flatten(), len(qwerty_rows[0])).numpy()
    y_train_one_hot = tf.one_hot(y_train.flatten(), len(q
    input_size = len(qwerty_rows[0])
    output_size = input_size
    model = create_neural_network(input_size, output_size)
    train_neural_network(model, X_train_one_hot, y_train_one_hot, epochs=10)

if __name__ == '__main__':
    main()

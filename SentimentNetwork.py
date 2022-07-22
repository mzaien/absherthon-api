import uvicorn
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import schemas
import crud
import joblib as jb
import pandas as pd
import time
import numpy as np

# The SentimentNetwork should reset here but for some reason python/fastapi could not read it from here 
# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, con, hidden_nodes=10, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reports(list) - List of reports used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reports and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),
                          hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):

        # populate review_vocab with all of the words in the given report
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        # populate label_vocab with all of the words in the given labels.
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def get_conf(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like in the "train" function.

        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        return layer_2[0][0]

    def get_pred(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like in the "train" function.

        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values

        if(layer_2[0] >= 0.6):
            return True

        else:
            return False

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5,
                                            (self.hidden_nodes, self.output_nodes))

        # The input layer, a two-dimensional matrix with shape 1 x input_nodes
        self.layer_0 = np.zeros((1, input_nodes))

    def update_input_layer(self, review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0

        for word in review.split(" "):
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] += 1

    def get_target_for_label(self, label):
        if(label == 1):
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews, training_labels):

        # make sure out we have a matching number of reports and labels
        assert(len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reports and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # Get the next report and its correct label
            review = training_reviews[i]
            label = training_labels[i]

            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            self.update_input_layer(review)

            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            # Output layer error is the difference between desired target and actual output.
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * \
                self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            # errors propagated to the hidden layer
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            # hidden layer gradients - no nonlinearity so it's the same as the error
            layer_1_delta = layer_1_error

            # Update the weights
            # update hidden-to-output weights with gradient descent step
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * \
                self.learning_rate
            # update input-to-hidden weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(
                layer_1_delta) * self.learning_rate

            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 1):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 0):
                correct_so_far += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            # sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
            #                  + "% Speed(reports/sec):" + str(reviews_per_second)[0:5] \
            #                  + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
            #                  + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels, testing_con):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i], testing_con)
            if(pred >= 0.6):
                print(testing_reviews[i]+": "+"بلاغ حقيقي")
                print("-----------------True log---------------")
                return {
                    "type": "بلاغ حقيقي",
                    "score": pred,
                    "text": testing_reviews[i]
                }
            else:
                print(testing_reviews[i]+": "+"بلاغ كاذب")
                print("---------------False log-------------")
                return {
                    "type": "بلاغ كاذب",
                    "score": pred,
                    "text": testing_reviews[i]
                }
            if(pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            # sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
            #                  + "% Speed(reports/sec):" + str(reviews_per_second)[0:5] \
            #                  + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
            #                  + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review, con):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like in the "train" function.

        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if(layer_2[0] >= 0.6):
            print("نسبة الثقة:", layer_2[0][0])
            return layer_2[0][0]

        else:
            print("نسبة الثقة:", layer_2[0][0])
            return layer_2[0][0]


con_validaiton = []


def check_report(report):
    mlp = jb.load('spam_classifier.joblib')
    report_df = pd.DataFrame(columns=['report_id', 'crime_type', 'social_app', '@suspect_account',
                             'account_status', 'crime_link', 'crime_description', 'ligit', 'conf'])
    report_description = {'report_id': 0, 'crime_type': '', 'social_app': '', '@suspect_account': '',
                          'account_status': '', 'crime_link': '', 'crime_description': report, 'ligit': 0, 'conf': 0}
    report_df = report_df.append(report_description, ignore_index=True)
    result = mlp.test(report_df['crime_description']
                        [0:1], report_df['ligit'][0:1], con_validaiton)
    return result


# intialize web app / pi
app = FastAPI()


# Allows cors for everyone **Ignore**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# Redirects base url to docs goto /redoc for fancy documentation
@app.get("/")
def main():
    return RedirectResponse(url="/docs")


# post request for Name Read
@app.post("/api/model", response_model=schemas.predictionResponse)
def check(body: schemas.Crime = Body(...)):
    print("crime", body.crime)
    result = check_report(body.crime)
    crud.insert(result)
    crud.list()
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

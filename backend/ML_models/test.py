# Import your SentimentAnalyzer class

import pandas as pd
from sentiment_analyzer import SentimentAnalyzer #--------------Modified-----------------------------------------------------------------

# from ensemble_analyzer import EnsembleSentimentAnalyzer


# The list of 50 test cases from above
test_suite = [
    # --- Positive Comments (Clear Support) ---
    {"text": "This amendment is a welcome move for the financial sector.", "true_sentiment": "Positive"}
   
]

def run_test():
    """
    Initializes the analyzer, runs the test suite, and prints the accuracy report.
    """
    print("--- Starting Model Accuracy Test ---")
    
    analyzer = SentimentAnalyzer() #--------------Modified-----------------------------------------------------------------
    # analyzer = EnsembleSentimentAnalyzer()


    correct_predictions = 0
    total_predictions = len(test_suite)
    
    print("\n--- Running Predictions ---")
    for i, test_case in enumerate(test_suite):
        comment = test_case["text"]
        true_sentiment = test_case["true_sentiment"]
        
        # Get the model's prediction
        result = analyzer.analyze(comment)
        predicted_sentiment = result.get("top_sentiment", "Error") #--------------Modified-----------------------------------------------------------------
        # predicted_sentiment = result.get("final_sentiment", "Error")

        # Compare and log the outcome
        is_correct = (predicted_sentiment == true_sentiment)
        if is_correct:
            correct_predictions += 1
        
        status = "✅ Correct" if is_correct else "❌ MISMATCH"
        print(f"#{i+1}: {status}")
        print(f"   - Comment: '{comment[:60]}...'")
        print(f"   - True: {true_sentiment}, Predicted: {predicted_sentiment}")
        if not is_correct:
            print(f"   - Full Model Output: {result.get('all_scores')}")  #--------------Modified-----------------------------------------------------------------
            # print(f"   - Specialist: {result.get('specialist_prediction')}, Generalist: {result.get('generalist_prediction')}")
        
        print("-" * 20)

    # Calculate and print the final accuracy score
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n--- Test Complete ---")
    print(f"Final Score: {correct_predictions} / {total_predictions} correct.")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    file  = open("C:\\Users\\Vansh\\Desktop\\SIH\\Project\\dummy.csv", "r")
    data = file.readlines()
    # for i in range(1,len(data)):
    #     string = data[i].split(",")
    #     dict = {}
    #     dict["text"] = string[3]
    #     dict["true_sentiment"] = string[10]
    #     test_suite.append(dict)
    

    # for j in test_suite:
    #     print(j['true_sentiment'])

    file = pd.read_csv("C:\\Users\\Vansh\\Desktop\\SIH\\Project\\dummy.csv")
    print(file.head())

    test = file[["CommentText", "single_value"]].values
    for i in test:
        
        dict = {}
        dict["text"] = i[0]
        dict["true_sentiment"] = i[1]
        test_suite.append(dict)

 
    run_test()



"""

----------------------------Finbert ----------------------------

--- Test Complete ---
Final Score: 37 / 45 correct.
Accuracy: 82.22%


----------------------------Ensemble Score is low ----------------------------- why ? bad logic -----------------

--- Test Complete ---
Final Score: 36 / 45 correct.
Accuracy: 80.00%
"""
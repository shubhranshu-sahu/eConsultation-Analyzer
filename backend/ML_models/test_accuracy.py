# Import your SentimentAnalyzer class

# from sentiment_analyzer import SentimentAnalyzer #--------------Modified-----------------------------------------------------------------

from ensemble_analyzer import EnsembleSentimentAnalyzer


# The list of 50 test cases from above
test_suite = [
    # --- Positive Comments (Clear Support) ---
    {"text": "This amendment is a welcome move for the financial sector.", "true_sentiment": "Positive"},
    {"text": "We strongly support the proposal to provide ease of doing business for Finance Companies in the IFSC.", "true_sentiment": "Positive"},
    {"text": "Excellent clarification in Rule 11(2); this will benefit many NBFCs.", "true_sentiment": "Positive"},
    {"text": "The ministry's decision to consult with stakeholders is commendable and leads to better regulations.", "true_sentiment": "Positive"},
    {"text": "This is a positive development and aligns with global standards.", "true_sentiment": "Positive"},
    {"text": "We agree with the justification provided and approve of the draft.", "true_sentiment": "Positive"},
    {"text": "The inclusion of IFSCA-registered firms is a logical and beneficial step.", "true_sentiment": "Positive"},
    {"text": "This change will undoubtedly foster growth in the IFSC jurisdiction.", "true_sentiment": "Positive"},
    {"text": "A well-drafted and necessary amendment. We offer our full support.", "true_sentiment": "Positive"},
    {"text": "This will streamline operations and reduce compliance overhead.", "true_sentiment": "Positive"},
    {"text": "Kudos to the MCA for this forward-thinking initiative.", "true_sentiment": "Positive"},
    {"text": "The proposed rule change is both practical and beneficial for the industry.", "true_sentiment": "Positive"},
    {"text": "We are pleased with the direction the Ministry is taking with this draft.", "true_sentiment": "Positive"},
    {"text": "This is a much-needed reform that we have been advocating for.", "true_sentiment": "Positive"},
    {"text": "The proposal is sound and the economic benefits are clear.", "true_sentiment": "Positive"},

    # --- Negative Comments (Clear Opposition/Concern) ---
    {"text": "This amendment introduces significant risks without adequate safeguards.", "true_sentiment": "Negative"},
    {"text": "We oppose this proposal as it creates an unfair advantage for certain companies.", "true_sentiment": "Negative"},
    {"text": "The draft is vague on the definition of 'Finance Company', which is a major concern.", "true_sentiment": "Negative"},
    {"text": "Extending this exemption is a mistake and weakens regulatory oversight.", "true_sentiment": "Negative"},
    {"text": "This is a step in the wrong direction and could lead to market instability.", "true_sentiment": "Negative"},
    {"text": "I am deeply concerned about the lack of a proper impact assessment.", "true_sentiment": "Negative"},
    {"text": "The potential for misuse of this exemption is alarmingly high.", "true_sentiment": "Negative"},
    {"text": "This rule change is detrimental to fair competition.", "true_sentiment": "Negative"},
    {"text": "We believe this proposal is ill-conceived and should be withdrawn immediately.", "true_sentiment": "Negative"},
    {"text": "The justification provided is weak and unconvincing.", "true_sentiment": "Negative"},
    {"text": "This will create a regulatory loophole that could be exploited.", "true_sentiment": "Negative"},
    {"text": "The burden on smaller NBFCs will increase as a result of this.", "true_sentiment": "Negative"},
    {"text": "We foresee negative consequences for the market if this is implemented.", "true_sentiment": "Negative"},
    {"text": "This is a poorly drafted amendment that fails to address key issues.", "true_sentiment": "Negative"},
    {"text": "I cannot support any measure that diminishes accountability.", "true_sentiment": "Negative"},

    # --- Neutral Comments (Factual, Questions, or Mild Suggestions) ---
    {"text": "We request clarification on the applicability of this rule to foreign subsidiaries.", "true_sentiment": "Neutral"},
    {"text": "The document states the deadline for comments is July 17th, 2025.", "true_sentiment": "Neutral"},
    {"text": "Could the Ministry provide the data that informed this proposal?", "true_sentiment": "Neutral"},
    {"text": "We suggest changing the word 'include' to 'comprise' in Section 2(a) for better clarity.", "true_sentiment": "Neutral"},
    {"text": "It is noted that this rule is an amendment to the Companies Rules, 2014.", "true_sentiment": "Neutral"},
    {"text": "Our firm will be submitting a detailed analysis before the deadline.", "true_sentiment": "Neutral"},
    {"text": "The notification references circular 1/32/2013-CLV(Part).", "true_sentiment": "Neutral"},
    {"text": "This proposal will affect all companies registered with the IFSCA.", "true_sentiment": "Neutral"},
    {"text": "We acknowledge receipt of the draft notification.", "true_sentiment": "Neutral"},
    {"text": "The effective date will be the date of its publication in the Official Gazette.", "true_sentiment": "Neutral"},

    # --- Tricky/Nuanced Comments ---
    {"text": "While we support the goal of ease of doing business, the execution here is flawed.", "true_sentiment": "Negative"}, # Mixed but leans negative
    {"text": "The proposal is not bad, but it could be significantly improved.", "true_sentiment": "Neutral"}, # Mild suggestion
    {"text": "This is just what not should be done.", "true_sentiment": "Negative"}, # The one that failed before
    {"text": "Sure, another exemption. What could possibly go wrong?", "true_sentiment": "Negative"}, # Sarcasm
    {"text": "This doesn't really change much for the established players in the market.", "true_sentiment": "Neutral"}, # Apathetic
]

def run_test():
    """
    Initializes the analyzer, runs the test suite, and prints the accuracy report.
    """
    print("--- Starting Model Accuracy Test ---")
    
    # analyzer = SentimentAnalyzer() #--------------Modified-----------------------------------------------------------------
    analyzer = EnsembleSentimentAnalyzer()


    correct_predictions = 0
    total_predictions = len(test_suite)
    
    print("\n--- Running Predictions ---")
    for i, test_case in enumerate(test_suite):
        comment = test_case["text"]
        true_sentiment = test_case["true_sentiment"]
        
        # Get the model's prediction
        result = analyzer.analyze(comment)
        # predicted_sentiment = result.get("top_sentiment", "Error") #--------------Modified-----------------------------------------------------------------
        predicted_sentiment = result.get("final_sentiment", "Error")

        # Compare and log the outcome
        is_correct = (predicted_sentiment == true_sentiment)
        if is_correct:
            correct_predictions += 1
        
        status = "✅ Correct" if is_correct else "❌ MISMATCH"
        print(f"#{i+1}: {status}")
        print(f"   - Comment: '{comment[:60]}...'")
        print(f"   - True: {true_sentiment}, Predicted: {predicted_sentiment}")
        if not is_correct:
            # print(f"   - Full Model Output: {result.get('all_scores')}")  #--------------Modified-----------------------------------------------------------------
            print(f"   - Specialist: {result.get('specialist_prediction')}, Generalist: {result.get('generalist_prediction')}")
        
        print("-" * 20)

    # Calculate and print the final accuracy score
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n--- Test Complete ---")
    print(f"Final Score: {correct_predictions} / {total_predictions} correct.")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
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
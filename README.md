# Question Answering Model

This repository contains code to train and evaluate a Question Answering (QA) model using the `simpletransformers` library with a BERT-based approach.

## Files in the Repository

- `question_answer (1).ipynb`: Jupyter Notebook containing the training and evaluation workflow for the QA model.
- `predictions (1).json`: JSON file storing the model's predictions.
- `requirements (1).txt`: List of dependencies required to run the code.
- `train (1).json`: Training dataset in SQuAD-like format.
- `test (1).json`: Test dataset in SQuAD-like format.

## Setup and Installation

1. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv qna_env
   source qna_env/bin/activate  # On Windows use `qna_env\Scripts\activate`
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements (1).txt
   ```

## Training the Model

Run the Jupyter Notebook `question_answer (1).ipynb` to train the model. The key steps include:
- Loading the dataset (`train (1).json`, `test (1).json`)
- Initializing a BERT-based model (`bert-base-cased`)
- Training with hyperparameter tuning
- Evaluating on test data

## Making Predictions

To make predictions using the trained model, provide a context and a question in the specified format:

```python
from simpletransformers.question_answering import QuestionAnsweringModel

model = QuestionAnsweringModel("bert", "outputs/bert/best_model")

to_predict = [{
    "context": "Vin is a Mistborn of great power and skills.",
    "qas": [{
        "question": "What is Vin's speciality?",
        "id": "0",
    }]
}]

answers, probabilities = model.predict(to_predict)
print(answers)
```

## Evaluation Metrics

During training, the model tracks:
- Loss per epoch
- Correct, similar, and incorrect predictions
- `eval_loss` for performance assessment

## Future Improvements

- Fine-tune with a larger dataset
- Experiment with different transformer models like RoBERTa or DistilBERT
- Implement data augmentation techniques

## License

This project is open-source and available for use under the MIT License.


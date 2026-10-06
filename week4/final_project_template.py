# Week 4 Starter Code

This folder contains a final project template for students.

## Final Project Template

```python
# week4/final_project_template.py

"""
Final project template for the AI course.

Students may adapt this to any topic:
- chatbot
- spam detector
- sentiment analysis
- image classifier
- digit recognition
"""


def problem_statement():
    print("Project title: My AI Project")
    print("Problem: I will build an AI model to classify or predict something.")


def collect_data():
    # Add data here
    sample_data = [
        ("example input 1", "label 1"),
        ("example input 2", "label 2"),
    ]
    return sample_data


def pre_process_data(data):
    # Prepare the dataset for training
    inputs = [item[0] for item in data]
    labels = [item[1] for item in data]
    return inputs, labels


def train_model(inputs, labels):
    # Replace with actual model code
    print("Training model with:")
    for item in zip(inputs, labels):
        print(item)
    print("Model trained successfully.")


def evaluate_model():
    # Add metrics here: accuracy, precision, confusion matrix, etc.
    print("Model evaluation results:")
    print("Accuracy: 0.00")


def main():
    problem_statement()
    data = collect_data()
    inputs, labels = pre_process_data(data)
    train_model(inputs, labels)
    evaluate_model()


if __name__ == "__main__":
    main()
```

## Final Project Checklist

- Write a short project description
- Name the AI problem you are solving
- Describe the data used
- Explain how the model is trained
- Show at least one result or prediction
- Discuss improvements and limitations

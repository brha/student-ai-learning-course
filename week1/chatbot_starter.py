# Week 1 Starter Code

This folder contains simple starter activities for Week 1.

## 1. Rule-Based Chatbot

This chatbot answers based on keywords.

```python
# week1/chatbot_starter.py

def chatbot_response(user_input: str) -> str:
    text = user_input.lower()

    if "hello" in text or "hi" in text:
        return "Hello! How can I help you today?"
    elif "school" in text:
        return "School is important. What would you like to know about it?"
    elif "ai" in text:
        return "AI stands for Artificial Intelligence. It helps computers learn patterns from data."
    elif "bye" in text or "goodbye" in text:
        return "Goodbye! See you next time."
    else:
        return "I am not sure how to answer that yet. Try asking about school, AI, or hello."


def main():
    print("AI Chatbot Demo")
    print("Type 'exit' to stop.\n")

    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user))


if __name__ == "__main__":
    main()
```

## 2. Simple Student Class Example

```python
# week1/student_class.py

class Student:
    def __init__(self, name, grade, subject):
        self.name = name
        self.grade = grade
        self.subject = subject

    def info(self):
        return f"{self.name} has grade {self.grade} in {self.subject}."


student1 = Student("Ana", "A", "AI")
print(student1.info())
```

## Challenge

- Add more keywords and responses
- Add a list of known questions and answers
- Make the bot answer based on a dictionary instead of if/else

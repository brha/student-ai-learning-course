## 2. Simple Student Class Example

#python
# week1/student_class.py

class Student:
    def __init__(self, name, grade, subject):
        self.name = name
        self.grade = grade
        self.subject = subject

    def __str__(self):
        return f"{self.name} has grade {self.grade} in {self.subject}."


students = []
students.append(Student("Ana", "A", "AI"))
students.append(Student("Bent", "B","AI"))
for student in students: print(student)

'''

## Challenge


- Add more keywords and responses
- Add a list of known questions and answers
- Make the bot answer based on a dictionary instead of if/else
'''
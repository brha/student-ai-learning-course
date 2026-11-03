# Lesson 13: A simple neuron analogy
# No extra libraries are needed.
#
# Visual activity: https://playground.tensorflow.org
# Read: https://developers.google.com/machine-learning/crash-course/neural-networks

# --------------------------------------------------
# Step 1: Choose input values between 0 and 1
# --------------------------------------------------
# Students: Change these values and see how the result changes.
hours_studied = 0.8
sleep = 0.6
attendance = 0.9

# --------------------------------------------------
# Step 2: Give each input a weight
# --------------------------------------------------
# A bigger weight means that input has more influence.
# Students: Change the weights.
hours_weight = 0.5
sleep_weight = 0.2
attendance_weight = 0.4

# --------------------------------------------------
# Step 3: Calculate the neuron's score
# --------------------------------------------------
# This is a simplified weighted sum.
bias = -0.7
score = (
    hours_studied * hours_weight
    + sleep * sleep_weight
    + attendance * attendance_weight
    + bias
)

print("Step 3: Neuron score:", round(score, 2))

# --------------------------------------------------
# Step 4: Apply a decision threshold
# --------------------------------------------------
# If the score is at least 0, the simplified neuron outputs 1.
# Students: Change the threshold and observe the decision.
threshold = 0

if score >= threshold:
    prediction = 1
else:
    prediction = 0

print("Step 4: Neuron output:", prediction)
print("Interpretation:", "PASS" if prediction == 1 else "NOT PASS")

# --------------------------------------------------
# Step 5: Experiment
# --------------------------------------------------
# Try changing one input or weight at a time.
# Which change has the biggest effect on the score?
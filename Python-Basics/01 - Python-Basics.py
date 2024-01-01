# Python Programming Fundamentals: An Extended Example

# --- BASIC VARIABLES AND DATA TYPES ---
print("\n--- BASIC VARIABLES AND DATA TYPES ---")
# Numeric Variables
int_value = 42
float_value = 3.14159

# String Variables
simple_string = "Hello, Python!"
multiline_string = """This is a multi-line
string example.
Defined using triple quotes in Python."""

# Boolean Variables
bool_value = True  # or False

# Printing Data Types
print("Integer:", int_value)
print("Float:", float_value)
print("String:", simple_string)
print("Multi-line String:", multiline_string)
print("Boolean:", bool_value)

# --- LISTS AND LIST OPERATIONS ---
print("\n--- LISTS AND LIST OPERATIONS ---")
# List Definition
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "cherry"]

# Accessing List Elements
print("First number:", numbers[0])
print("Second fruit:", fruits[1])

# Modifying Lists
numbers[2] = 10
print("Updated Numbers List:", numbers)

# Adding Elements to List
fruits.append("orange")
print("Updated Fruits List:", fruits)

# Deleting Elements from List
del numbers[1]
print("Numbers after Deletion:", numbers)

# Looping Through a List
for fruit in fruits:
    print(fruit)

# --- DICTIONARIES AND DICTIONARY OPERATIONS ---
print("\n--- DICTIONARIES AND DICTIONARY OPERATIONS ---")
# Dictionary Definition
people = {
    "Ahmet": 25,
    "Ayşe": 30,
    "Mehmet": 35
}

# Accessing Dictionary Elements
print("Ahmet's age:", people["Ahmet"])

# Modifying Dictionaries
people["Ayşe"] = 32
print("Updated People Dictionary:", people)

# Adding Elements to Dictionary
people["Ali"] = 28
print("Dictionary after Adding a New Person:", people)

# Deleting Elements from Dictionary
del people["Mehmet"]
print("Dictionary after Deletion:", people)

# Looping Through a Dictionary
for name, age in people.items():
    print(f"{name}'s age is: {age}")

# --- CONDITIONAL STATEMENTS AND CONTROL FLOW ---
print("\n--- CONDITIONAL STATEMENTS AND CONTROL FLOW ---")
# Basic Conditional Statement
number = 15
if number > 10:
    print("Number is greater than 10.")
elif number < 10:
    print("Number is less than 10.")
else:
    print("Number is 10.")

# Logical Operators
if number > 10 and number < 20:
    print("Number is between 10 and 20.")

# --- LOOPS ---
print("\n--- LOOPS ---")
# For Loop
for i in range(5):
    print(i)

# While Loop
i = 0
while i < 5:
    print(i)
    i += 1

# --- FUNCTIONS ---
print("\n--- FUNCTIONS ---")
# Function Definition and Calling
def add(a, b):
    return a + b

result = add(3, 5)
print("Sum:", result)

# Function with Optional Parameters
def greet(name="Guest"):
    return f"Hello, {name}!"

print(greet("Ahmet"))
print(greet())  # Using the optional parameter

# --- ERROR HANDLING AND EXCEPTIONS ---
print("\n--- ERROR HANDLING AND EXCEPTIONS ---")
# Try-Except Blocks
try:
    # Code that might cause an error
    result = 10 / 0
except ZeroDivisionError:
    # Handling the error
    print("Cannot divide by zero!")

# --- FILE OPERATIONS ---
print("\n--- FILE OPERATIONS ---")
# Writing to a File
file_path = "example_file.txt"
with open(file_path, "w") as file:
    file.write("File operations in Python.\n")

# Reading from a File
with open(file_path, "r") as file:
    content = file.read()
    print("File Content:", content)

# --- EXTRA: LIST COMPREHENSIONS ---
print("\n--- EXTRA: LIST COMPREHENSIONS ---")
# List Comprehension Example
squares = [x * x for x in range(10)]
print("Squares of Numbers:", squares)

# Conditional List Comprehension
even_squares = [x * x for x in range(10) if x % 2 == 0]
print("Squares of Even Numbers:", even_squares)

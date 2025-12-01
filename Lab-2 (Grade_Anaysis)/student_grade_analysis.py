# ------------------------------------------------------------
#  GradeBook Analyzer CLI
# ------------------------------------------------------------
import csv
from statistics import median


def print_welcome():
    print("==============================================")
    print(" Welcome to the GradeBook Analyzer CLI ")
    print("==============================================")
    print("This tool analyzes student marks and assigns grades.\n")
    print("Please choose an input method:")
    print("1. Manual entry of student names and marks")
    print("2. Load data from a CSV file\n")


def load_data():
    marks = {}
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        num_students = int(input("\nHow many students? "))
        for i in range(num_students):
            name = input(f"Enter name of student {i+1}: ")
            mark = float(input(f"Enter marks for {name}: "))
            marks[name] = mark
    elif choice == "2":
        filename = input("Enter CSV filename (with .csv extension): ")
        try:
            with open(filename, "r") as file:
                reader = csv.reader(file)
                next(reader)  
                for row in reader:
                    name, mark = row
                    marks[name] = float(mark)
            print("\n Data loaded successfully from CSV!")
        except FileNotFoundError:
            print(" File not found. Please check the filename.")
            return {}
    else:
        print("Invalid choice. Exiting program.")
        return {}
    return marks



def calculate_average(marks_dict):
    return sum(marks_dict.values()) / len(marks_dict)

def calculate_median(marks_dict):
    return median(marks_dict.values())

def find_max_score(marks_dict):
    return max(marks_dict.values())

def find_min_score(marks_dict):
    return min(marks_dict.values())

def print_statistics(marks):
    print("\n--- Statistical Summary ---")
    print(f"Average Marks: {calculate_average(marks):.2f}")
    print(f"Median Marks:  {calculate_median(marks):.2f}")
    print(f"Highest Marks: {find_max_score(marks):.2f}")
    print(f"Lowest Marks:  {find_min_score(marks):.2f}")


def assign_grades(marks):
    grades = {}
    for name, mark in marks.items():
        if mark >= 90:
            grade = "A"
        elif mark >= 80:
            grade = "B"
        elif mark >= 70:
            grade = "C"
        elif mark >= 60:
            grade = "D"
        else:
            grade = "F"
        grades[name] = grade
    return grades

def grade_distribution(grades):
    distribution = {g: list(grades.values()).count(g) for g in ["A", "B", "C", "D", "F"]}
    print("\n--- Grade Distribution ---")
    for g, count in distribution.items():
        print(f"Grade {g}: {count} student(s)")


def pass_fail_summary(marks):
    passed_students = [name for name, mark in marks.items() if mark >= 40]
    failed_students = [name for name, mark in marks.items() if mark < 40]

    print("\n--- Pass/Fail Summary ---")
    print(f"Passed Students ({len(passed_students)}): {', '.join(passed_students) if passed_students else 'None'}")
    print(f"Failed Students ({len(failed_students)}): {', '.join(failed_students) if failed_students else 'None'}")


def display_results_table(marks, grades):
    print("\nName\t\tMarks\tGrade")
    print("---------------------------------")
    for name in marks:
        print(f"{name:<15}\t{marks[name]:<6.2f}\t{grades[name]}")
    print("---------------------------------")

def main():
    while True:
        print_welcome()
        marks = load_data()

        if not marks:
            print("No data to analyze. Exiting...")
            break

        print_statistics(marks)
        grades = assign_grades(marks)
        grade_distribution(grades)
        pass_fail_summary(marks)
        display_results_table(marks, grades)

        again = input("\nDo you want to analyze another dataset? (yes/no): ").strip().lower()
        if again != "yes":
            print("\nThank you for using the GradeBook Analyzer!")
            break



if __name__ == "__main_1_":
    main()
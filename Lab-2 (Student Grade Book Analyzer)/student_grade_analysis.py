"""
gradebook.py
Name      : Tanish
Date      : 01-12-2025
Title     : Gradebook Analyzer
"""

import csv


# -------------------- General Welcome & Menus -------------------- #

def print_welcome():
    print("\n" + "=" * 50)
    print("          Welcome to the Gradebook Analyzer")
    print("=" * 50)
    print("This program analyzes student marks using:")
    print("  - Manual input")
    print("  - CSV file (.csv)")
    print("-" * 50)


def print_input_menu():
    print("\nChoose data input method:")
    print("1) Manual entry of student names and marks")
    print("2) Load from CSV file (.csv)")
    print("q) Quit program")


def print_analysis_menu():
    print("\nWhat would you like to do with this dataset?")
    print("1) Statistical summary (average, median, highest, lowest)")
    print("2) Grade assignment and distribution (A–F)")
    print("3) Pass/Fail filter using list comprehension")
    print("4) Results table (with option to save as CSV)")
    print("q) Quit analysis and exit program")


# -------------------- Data Entry / CSV Import -------------------- #

def get_manual_marks():
    """
    Allows the user to manually enter student names and marks.
    Returns a dictionary: {name: marks}
    """
    marks_dict = {}

    while True:
        try:
            count = int(input("\nEnter number of students: "))
            if count <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    for i in range(1, count + 1):
        print(f"\nStudent {i}:")
        name = input("  Enter name  : ").strip()
        while True:
            try:
                mark = float(input("  Enter marks : "))
                if mark < 0:
                    print("  Marks cannot be negative. Try again.")
                    continue
                break
            except ValueError:
                print("  Invalid marks. Please enter a number.")
        marks_dict[name] = mark

    return marks_dict


def get_marks_from_csv():
    """
    Loads student names and marks from a CSV file.
    Assumes each row is: name, marks
    First row may be a header.
    Returns a dictionary: {name: marks}
    """
    marks_dict = {}
    filename = input("\nEnter CSV file path (e.g., data.csv): ").strip()

    try:
        with open(filename, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue  # skip empty rows

                if len(row) < 2:
                    print(f"Skipping row {i + 1}: not enough columns.")
                    continue

                name_raw = row[0].strip()
                marks_raw = row[1].strip()

                try:
                    marks_value = float(marks_raw)
                except ValueError:
                    # Probably header or bad row
                    if i == 0:
                        continue
                    print(f"Skipping row {i + 1}: invalid marks '{marks_raw}'.")
                    continue

                marks_dict[name_raw] = marks_value

        if not marks_dict:
            print("No valid data loaded from CSV.")
        else:
            print(f"Loaded {len(marks_dict)} records from '{filename}'.")
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return marks_dict


# -------------------- Statistical Analysis Functions -------------------- #

def calculate_average(marks_dict):
    if not marks_dict:
        return 0
    total = sum(marks_dict.values())
    return total / len(marks_dict)


def calculate_median(marks_dict):
    if not marks_dict:
        return 0
    scores = sorted(marks_dict.values())
    n = len(scores)
    mid = n // 2

    if n % 2 == 1:
        return scores[mid]
    else:
        return (scores[mid - 1] + scores[mid]) / 2


def find_max_score(marks_dict):
    if not marks_dict:
        return None
    return max(marks_dict.values())


def find_min_score(marks_dict):
    if not marks_dict:
        return None
    return min(marks_dict.values())


# -------------------- Grade Assignment & Distribution -------------------- #

def get_letter_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def assign_grades(marks_dict):
    """
    Returns:
      grades_dict: {name: grade}
      grade_distribution: { "A": count, "B": count, ... }
    """
    grades_dict = {}
    grade_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

    for name, score in marks_dict.items():
        grade = get_letter_grade(score)
        grades_dict[name] = grade
        grade_distribution[grade] += 1

    return grades_dict, grade_distribution


# -------------------- Pass / Fail Filter (List Comprehension) -------------------- #

def get_pass_fail_lists(marks_dict):
    passed_students = [name for name, score in marks_dict.items() if score >= 40]
    failed_students = [name for name, score in marks_dict.items() if score < 40]
    return passed_students, failed_students


# -------------------- Results Table & CSV Export -------------------- #

def print_results_table(marks_dict, grades_dict):
    print("\nResults Table:")
    print("Name\t\tMarks\tGrade")
    print("----------------------------------------")
    for name in sorted(marks_dict.keys()):
        marks = marks_dict[name]
        grade = grades_dict.get(name, "-")
        print(f"{name:<10}\t{marks:<6.2f}\t{grade}")


def export_results_to_csv(marks_dict, grades_dict, filename="results.csv"):
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Marks", "Grade", "Status"])
            for name, marks in marks_dict.items():
                grade = grades_dict.get(name, "-")
                status = "Pass" if marks >= 40 else "Fail"
                writer.writerow([name, marks, grade, status])
        print(f"\nResults saved successfully to '{filename}'.")
    except Exception as e:
        print(f"\nError while saving CSV: {e}")


# -------------------- Option Handlers (1–4) -------------------- #

def option_1_statistical_summary(marks_dict):
    print("\n--- Option 1: Statistical Summary ---")
    if not marks_dict:
        print("No data available.")
        return

    avg = calculate_average(marks_dict)
    median = calculate_median(marks_dict)
    max_score = find_max_score(marks_dict)
    min_score = find_min_score(marks_dict)

    print(f"Total students : {len(marks_dict)}")
    print(f"Average score  : {avg:.2f}")
    print(f"Median score   : {median:.2f}")
    print(f"Highest score  : {max_score:.2f}")
    print(f"Lowest score   : {min_score:.2f}")


def option_2_grade_distribution(marks_dict):
    print("\n--- Option 2: Grade Assignment and Distribution ---")
    if not marks_dict:
        print("No data available.")
        return

    grades_dict, grade_distribution = assign_grades(marks_dict)

    print("\nGradebook (Name -> Grade):")
    for name in sorted(grades_dict.keys()):
        print(f"{name:<10} -> {grades_dict[name]}")

    print("\nGrade Distribution (A–F):")
    for grade in ["A", "B", "C", "D", "F"]:
        print(f"{grade}: {grade_distribution[grade]}")


def option_3_pass_fail_filter(marks_dict):
    print("\n--- Option 3: Pass/Fail Filter (List Comprehension) ---")
    if not marks_dict:
        print("No data available.")
        return

    passed_students, failed_students = get_pass_fail_lists(marks_dict)

    print(f"Passed students ({len(passed_students)}): "
          f"{', '.join(passed_students) if passed_students else 'None'}")
    print(f"Failed students ({len(failed_students)}): "
          f"{', '.join(failed_students) if failed_students else 'None'}")


def option_4_results_table_and_loop(marks_dict):
    print("\n--- Option 4: Results Table and User Loop ---")
    if not marks_dict:
        print("No data available.")
        return

    grades_dict, _ = assign_grades(marks_dict)
    print_results_table(marks_dict, grades_dict)

    # Ask if user wants to save to CSV
    save_choice = input("\nDo you want to save this table to 'results.csv'? (y/n): ").strip().lower()
    if save_choice == "y":
        export_results_to_csv(marks_dict, grades_dict)
    else:
        print("CSV export skipped.")


# -------------------- Main Program (User Loop) -------------------- #

def main():
    print_welcome()

    # First: choose input method and load data
    marks_dict = {}

    while not marks_dict:
        print_input_menu()
        choice = input("Enter your choice (1/2/q): ").strip().lower()

        if choice == "1":
            marks_dict = get_manual_marks()
        elif choice == "2":
            marks_dict = get_marks_from_csv()
        elif choice == "q":
            print("\nExiting Gradebook Analyzer. Goodbye!\n")
            return
        else:
            print("Invalid choice. Please select 1, 2, or q.")

        if not marks_dict and choice in ("1", "2"):
            print("No data loaded. Please try again.")

    # Now: analysis menu loop (options 1–4)
    while True:
        print_analysis_menu()
        option = input("Enter your choice (1–4 or 'q' to quit): ").strip().lower()

        if option == "1":
            option_1_statistical_summary(marks_dict)
        elif option == "2":
            option_2_grade_distribution(marks_dict)
        elif option == "3":
            option_3_pass_fail_filter(marks_dict)
        elif option == "4":
            option_4_results_table_and_loop(marks_dict)
        elif option == "q":
            print("\nThank you for using Gradebook Analyzer. Goodbye!\n")
            break
        else:
            print("Invalid choice. Please select 1–4 or 'q'.")


if __name__ == "__main__":
    main()

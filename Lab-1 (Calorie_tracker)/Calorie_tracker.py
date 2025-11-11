"""
---------------------------------------------------------
 Code  : Tanish Kumar
 Date    : 26 Sept 2025
 Project : Daily Calorie Tracker CLI
 Purpose : A console-based app to track meals and calories
---------------------------------------------------------
"""

import datetime
import os

# -----------------------------
# Predefined Calorie Chart 🍏
# -----------------------------
calorie_chart = {
    "Apple": 95,
    "Banana": 105,
    "Boiled Egg": 78,
    "Rice (1 cup)": 200,
    "Roti": 120,
    "Dal (1 cup)": 180,
    "Paneer (100g)": 265,
    "Chicken Breast (100g)": 165,
    "Milk (1 glass)": 150,
    "Bread Slice": 70,
    "Burger": 354,
    "Pizza Slice": 285,
    "French Fries (medium)": 365
}

# -----------------------------
# Fancy Welcome Banner 🎉
# -----------------------------
print("\n" + "="*60)
print("🌟✨ DAILY CALORIE TRACKER CLI ✨🌟".center(60))
print("="*60)
print(r"""
      🥗   🍎   🍔   🍕   🥚   🍗   🥛
   Track your meals, count your calories, 
     and stay within your daily goals!  
""")
print("="*60 + "\n")

# -----------------------------
# Input & Data Collection
# -----------------------------
meals = []
calories = []

num_meals = int(input("👉 How many meals do you want to enter today? "))

for i in range(num_meals):
    meal_name = input(f"\nEnter meal {i+1} name 🍽️ : ").capitalize()

    while True:
        choice = input(f"Enter calories for {meal_name} OR press 'c' to see chart 🔥: ").lower()
        
        if choice == 'c':
            print("\n📖 Food Calorie Chart:")
            print("-"*40)
            for food, cal in calorie_chart.items():
                print(f"{food:<25} {cal} cal")
            print("-"*40)
        else:
            try:
                cal_amount = float(choice)
                break
            except ValueError:
                print("Invalid input. Enter a number or press 'c' to check chart.")
    
    meals.append(meal_name)
    calories.append(cal_amount)

# -----------------------------
# Calorie Calculations
# -----------------------------
total_calories = sum(calories)
avg_calories = total_calories / len(calories)

daily_limit = float(input("\n Enter your daily calorie limit: "))

if total_calories > daily_limit:
    status_msg = f"You exceeded your limit by {total_calories - daily_limit:.2f} calories!"
else:
    status_msg = f"Great! You are within your limit. You have {daily_limit - total_calories:.2f} calories left."

# -----------------------------
# Neatly Formatted Output
# -----------------------------
print("\n" + "="*60)
print("📊 DAILY CALORIE SUMMARY 📊".center(60))
print("="*60)

print(f"{'Meal Name':<25}{'Calories'}")
print("-"*60)
for meal, cal in zip(meals, calories):
    print(f"{meal:<25}{cal}")
print("-"*60)
print(f"{'TOTAL':<25}{total_calories}")
print(f"{'AVERAGE':<25}{avg_calories:.2f}")
print("-"*60)
print(status_msg)
print("="*60)

# -----------------------------
# Save Session Log to File
# -----------------------------
save_choice = input("\n💾 Do you want to save this session log? (yes/no): ").lower()

if save_choice == "yes":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"calorie_log_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"Calorie Tracker Log - {timestamp}\n")
        f.write("="*60 + "\n")
        f.write(f"{'Meal Name':<25}{'Calories'}\n")
        f.write("-"*60 + "\n")
        for meal, cal in zip(meals, calories):
            f.write(f"{meal:<25}{cal}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'TOTAL':<25}{total_calories}\n")
        f.write(f"{'AVERAGE':<25}{avg_calories:.2f}\n")
        f.write(status_msg + "\n")
        f.write("="*60 + "\n")
    
    print(f"\n✅ Session saved successfully as '{filename}'.")

# -----------------------------
# View Previous Logs
# -----------------------------
view_choice = input("\n📂 Do you want to view previous logs? (yes/no): ").lower()

if view_choice == "yes":
    files = [f for f in os.listdir() if f.startswith("calorie_log_")]
    
    if not files:
        print("\n⚠️ No previous logs found.")
    else:
        print("\n📁 Available Logs:")
        for idx, file in enumerate(files, start=1):
            print(f"{idx}. {file}")
        
        try:
            select = int(input("\n👉 Enter the number of the log to open: "))
            if 1 <= select <= len(files):
                with open(files[select-1], "r", encoding="utf-8") as f:
                    print("\n" + "="*60)
                    print(f"📖 Viewing Log: {files[select-1]}".center(60))
                    print("="*60)
                    print(f.read())
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

print("\n🎉 Thank you for using Daily Calorie Tracker! Stay healthy 💪")

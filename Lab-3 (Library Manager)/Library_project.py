"""
Library Manager – Single File Version

Covers:
Task 1: Book class
Task 2: LibraryInventory manager
Task 3: JSON file persistence
Task 4: Menu-driven CLI
Task 5: Exception handling + logging
(All merged into one file for easier running)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

# -------------------- Logging Setup (Task 5) -------------------- #

LOG_FILE = "library.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# -------------------- Task 1: Book Class -------------------- #

class Book:
    """
    Represents a single book in the library.
    status: "available" or "issued"
    """

    def __init__(self, title: str, author: str, isbn: str, status: str = "available"):
        self.title = title.strip()
        self.author = author.strip()
        self.isbn = str(isbn).strip()
        self.status = status.strip().lower() if status else "available"

    def __str__(self) -> str:
        return f"[{self.isbn}] {self.title} by {self.author} - {self.status.capitalize()}"

    def to_dict(self) -> dict:
        """Convert Book object to dictionary (for JSON saving)."""
        return {
            "title": self.title,
            "author": self.author,
            "isbn": self.isbn,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Book":
        """Create a Book object from a dictionary."""
        return cls(
            title=data.get("title", ""),
            author=data.get("author", ""),
            isbn=data.get("isbn", ""),
            status=data.get("status", "available"),
        )

    # required methods:

    def issue(self) -> None:
        """Mark the book as issued if it is available."""
        if not self.is_available():
            raise ValueError("Book is already issued.")
        self.status = "issued"

    def return_book(self) -> None:
        """Mark the book as available."""
        if self.is_available():
            return
        self.status = "available"

    def is_available(self) -> bool:
        """Check if the book is available."""
        return self.status == "available"


# -------------------- Task 2 & 3: Inventory Manager -------------------- #

class LibraryInventory:
    """
    Manages a collection of Book objects and handles
    saving/loading them to/from a JSON file.
    """

    def __init__(self, storage_path: Path | str = "catalog.json"):
        self.storage_path = Path(storage_path)
        self.books: List[Book] = []
        self.load()  # load existing catalog if present

    # ----- core book operations (Task 2) -----

    def add_book(self, book: Book) -> None:
        """Add a new book to inventory."""
        self.books.append(book)
        logging.info("Book added: %s", book)
        self.save()

    def search_by_title(self, title: str) -> List[Book]:
        """Search for books whose title contains the given text (case-insensitive)."""
        title = title.lower().strip()
        return [b for b in self.books if title in b.title.lower()]

    def search_by_isbn(self, isbn: str) -> Optional[Book]:
        """Search for a book with the exact ISBN."""
        isbn = isbn.strip()
        for b in self.books:
            if b.isbn == isbn:
                return b
        return None

    def display_all(self) -> List[Book]:
        """Return a list of all books (for CLI printing)."""
        return list(self.books)

    # ----- JSON persistence (Task 3) -----

    def save(self) -> None:
        """Save the book catalog to JSON file."""
        try:
            data = [book.to_dict() for book in self.books]
            with self.storage_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logging.info("Catalog successfully saved to %s", self.storage_path)
        except (OSError, TypeError) as e:
            logging.error("Failed to save catalog: %s", e)
        finally:
            # just to show use of finally for file ops (Task 5)
            pass

    def load(self) -> None:
        """Load the book catalog from JSON file (if it exists)."""
        try:
            if not self.storage_path.exists():
                logging.info("Catalog file not found, starting with empty inventory.")
                self.books = []
                return

            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid catalog format (expected a list).")

            self.books = [Book.from_dict(d) for d in data]
            logging.info(
                "Catalog loaded from %s with %d books.",
                self.storage_path,
                len(self.books),
            )

        except FileNotFoundError:
            logging.error("Catalog file missing. Starting with empty inventory.")
            self.books = []
        except json.JSONDecodeError as e:
            logging.error("Catalog file is corrupted: %s. Starting empty.", e)
            self.books = []
        except Exception as e:
            logging.error("Unexpected error while loading catalog: %s", e)
            self.books = []
        finally:
            # again, finally used to complete the try-except structure
            pass


# -------------------- CLI Helper Functions -------------------- #

def get_non_empty_input(prompt: str) -> str:
    """Ask user until a non-empty string is entered. (Task 4 input validation)"""
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            print("\nInput cancelled.")
            return ""
        if value:
            return value
        print("Input cannot be empty. Please try again.")


def print_book_list(books: List[Book]) -> None:
    if not books:
        print("\nNo books found.\n")
    else:
        print("\nBooks:")
        for idx, book in enumerate(books, start=1):
            print(f"{idx}. {book}")
        print()  # blank line


# -------------------- CLI Actions (Task 4) -------------------- #

def add_book_cli(inventory: LibraryInventory) -> None:
    try:
        print("\n--- Add New Book ---")
        title = get_non_empty_input("Title: ")
        if not title:
            return
        author = get_non_empty_input("Author: ")
        if not author:
            return
        isbn = get_non_empty_input("ISBN: ")
        if not isbn:
            return

        # check duplicate ISBN
        if inventory.search_by_isbn(isbn):
            print("A book with this ISBN already exists.\n")
            return

        book = Book(title=title, author=author, isbn=isbn)
        inventory.add_book(book)
        print("Book added successfully.\n")

    except Exception as e:
        print(f"Error while adding book: {e}")
        logging.error("Error in add_book_cli: %s", e)


def issue_book_cli(inventory: LibraryInventory) -> None:
    try:
        print("\n--- Issue Book ---")
        isbn = get_non_empty_input("Enter ISBN of book to issue: ")
        if not isbn:
            return
        book = inventory.search_by_isbn(isbn)
        if not book:
            print("No book found with that ISBN.\n")
            return

        try:
            book.issue()
            inventory.save()
            print("Book issued successfully.\n")
            logging.info("Book issued: %s", book)
        except ValueError as e:
            print(str(e) + "\n")
            logging.warning("Issue attempt failed for %s: %s", book, e)

    except Exception as e:
        print(f"Error while issuing book: {e}")
        logging.error("Error in issue_book_cli: %s", e)


def return_book_cli(inventory: LibraryInventory) -> None:
    try:
        print("\n--- Return Book ---")
        isbn = get_non_empty_input("Enter ISBN of book to return: ")
        if not isbn:
            return
        book = inventory.search_by_isbn(isbn)
        if not book:
            print("No book found with that ISBN.\n")
            return

        book.return_book()
        inventory.save()
        print("Book returned successfully.\n")
        logging.info("Book returned: %s", book)

    except Exception as e:
        print(f"Error while returning book: {e}")
        logging.error("Error in return_book_cli: %s", e)


def view_all_cli(inventory: LibraryInventory) -> None:
    try:
        print("\n--- All Books ---")
        books = inventory.display_all()
        print_book_list(books)
    except Exception as e:
        print(f"Error while displaying books: {e}")
        logging.error("Error in view_all_cli: %s", e)


def search_cli(inventory: LibraryInventory) -> None:
    try:
        print("\n--- Search Books ---")
        print("1. Search by Title")
        print("2. Search by ISBN")
        choice = input("Enter choice (1/2): ").strip()

        if choice == "1":
            title = get_non_empty_input("Enter title text: ")
            if not title:
                return
            books = inventory.search_by_title(title)
            print_book_list(books)

        elif choice == "2":
            isbn = get_non_empty_input("Enter ISBN: ")
            if not isbn:
                return
            book = inventory.search_by_isbn(isbn)
            print_book_list([book] if book else [])

        else:
            print("Invalid choice.\n")

    except Exception as e:
        print(f"Error while searching: {e}")
        logging.error("Error in search_cli: %s", e)


# -------------------- Main Menu Loop (Task 4 & 5) -------------------- #

def main() -> None:
    catalog_path = Path("catalog.json")
    inventory = LibraryInventory(catalog_path)

    while True:
        print("=== Library Manager ===")
        print("1. Add Book")
        print("2. Issue Book")
        print("3. Return Book")
        print("4. View All Books")
        print("5. Search Books")
        print("6. Exit")

        try:
            choice = input("Enter your choice (1-6): ").strip()
        except EOFError:
            print("\nInput ended. Exiting.")
            break

        try:
            if choice == "1":
                add_book_cli(inventory)
            elif choice == "2":
                issue_book_cli(inventory)
            elif choice == "3":
                return_book_cli(inventory)
            elif choice == "4":
                view_all_cli(inventory)
            elif choice == "5":
                search_cli(inventory)
            elif choice == "6":
                print("Exiting Library Manager. Goodbye!")
                break
            else:
                print("Invalid choice. Please select from 1 to 6.\n")
        except Exception as e:
            # global safety net for unexpected errors
            print(f"An unexpected error occurred: {e}")
            logging.error("Unexpected error in main loop: %s", e)


if __name__ == "__main__":
    main()

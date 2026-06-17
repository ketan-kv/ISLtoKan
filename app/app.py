"""
ISL to Kannada Translator

Application Entry Point
"""

from app.menu import show_main_menu


def main_app():
    """Main application launcher."""

    while True:
        choice = show_main_menu()

        if choice == "1":
            print("\nStarting translation...\n")

            # Import only when the user starts translation
            import main
            main.main()

        elif choice == "2":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.\n")


if __name__ == "__main__":
    main_app()
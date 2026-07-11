import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    # Determine if running as bundled executable
    if getattr(sys, 'frozen', False):
        current_dir = sys._MEIPASS
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))

    # Change this to your actual main Streamlit filename
    # Options: 'app.py', 'main.py', 'streamlit_app.py'
    main_file = 'app.py'  # ← CHANGE THIS TO YOUR MAIN FILE NAME

    file_path = os.path.join(current_dir, main_file)

    # Verify the file exists
    if not os.path.exists(file_path):
        print(f"ERROR: Cannot find {file_path}")
        print(f"Looking in: {current_dir}")
        print("Available files:", os.listdir(current_dir))
        sys.exit(1)

    # Set Streamlit arguments
    sys.argv = [
        "streamlit", "run", file_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]

    sys.exit(stcli.main())
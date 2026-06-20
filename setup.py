"""
One-click project setup
Run this to set up everything automatically
"""
from utils.setup_helper import SetupHelper

if __name__ == "__main__":
    setup = SetupHelper()
    setup.setup_project()
    
    print("\n🎉 Project is ready!")
    print("\nNext step: Run the app")
    print("   streamlit run app.py")
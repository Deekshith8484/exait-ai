"""Quick launch script for EXRT AI applications."""

import subprocess
import sys

def main():
    print("EXRT AI - Application Launcher")
    print("=" * 50)
    print("\nAvailable Applications:")
    print("1. Main ECG Analyzer (File Upload)")
    print("2. Live Heart Monitor (Real-time Simulator)")
    print("3. Test ECG Generator")
    print("4. Exit")
    
    choice = input("\nSelect application (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Launching Main ECG Analyzer...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    
    elif choice == "2":
        print("\n🚀 Launching Live Heart Monitor...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "simulator/live_monitor.py"])
    
    elif choice == "3":
        print("\n🧪 Testing ECG Generator...")
        subprocess.run([sys.executable, "simulator/ecg_generator.py"])
    
    elif choice == "4":
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("\n❌ Invalid choice. Please select 1-4.")
        main()

if __name__ == "__main__":
    main()

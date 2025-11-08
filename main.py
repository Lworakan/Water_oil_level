#!/usr/bin/env python3
"""
Oil Classification and Volume Estimation System
Main entry point

Author: Senior Programmer
Date: 2025
"""

import tkinter as tk
from gui_components import OilClassifierGUI


def main():
    """Main application entry point"""
    print("=" * 70)
    print("Oil Classification & Volume Estimation System")
    print("=" * 70)
    print("\nInitializing application...")

    # Create root window
    root = tk.Tk()

    # Create application (keep reference to prevent garbage collection)
    app = OilClassifierGUI(root)
    root.app = app  # Store reference in root to prevent GC

    print("\nApplication started successfully!")
    print("\nHow to use:")
    print("  1. Place oil sample in the green detection area")
    print("  2. System will automatically classify oil quality and measure volume")
    print("  3. White color = No oil/Water")
    print("  4. Other colors (Yellow/Orange/Brown/Dark) = Oil detected")
    print("  5. Volume is calculated based on oil pixels vs empty space")
    print("\nPress 'Stop' to pause, 'Quit' to exit")
    print("=" * 70)

    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    main()

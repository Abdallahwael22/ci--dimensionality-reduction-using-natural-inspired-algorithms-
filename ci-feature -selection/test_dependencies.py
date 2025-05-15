def test_dependencies():
    """Test if all required dependencies are installed and working."""
    try:
        import tkinter
        print("✓ tkinter is working")
    except Exception as e:
        print(f"✗ tkinter error: {str(e)}")
        
    try:
        import numpy
        print("✓ numpy is working")
    except Exception as e:
        print(f"✗ numpy error: {str(e)}")
        
    try:
        import pandas
        print("✓ pandas is working")
    except Exception as e:
        print(f"✗ pandas error: {str(e)}")
        
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        print("✓ matplotlib is working")
    except Exception as e:
        print(f"✗ matplotlib error: {str(e)}")
        
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        print("✓ scikit-learn is working")
    except Exception as e:
        print(f"✗ scikit-learn error: {str(e)}")

if __name__ == "__main__":
    print("Testing dependencies...")
    test_dependencies()
    print("\nIf all tests passed (✓), you can run the main program.")
    print("If any test failed (✗), please install the missing dependencies:") 
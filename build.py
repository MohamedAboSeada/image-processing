import PyInstaller.__main__
import os
import sys
import shutil

def clean_build_dirs():
    """Clean build and dist directories"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    if os.path.exists('ImageEditor.spec'):
        os.remove('ImageEditor.spec')

def main():
    # Clean previous build
    clean_build_dirs()
    
    # Get the absolute path to the assets directory
    assets_path = os.path.abspath('assets')
    
    # Ensure assets directory exists
    if not os.path.exists(assets_path):
        print("Error: assets directory not found!")
        sys.exit(1)
    
    # Build the executable
    PyInstaller.__main__.run([
        'main.py',
        '--name=ImageEditor',
        '--onefile',
        '--windowed',
        '--icon=assets/logo.ico',
        f'--add-data={assets_path};assets',
        '--hidden-import=customtkinter',
        '--hidden-import=cv2',
        '--hidden-import=PIL',
        '--hidden-import=numpy',
        '--clean',
        '--noconfirm'
    ])
    
    print("\nBuild completed successfully!")
    print("The executable can be found in the 'dist' directory.")

if __name__ == '__main__':
    main() 
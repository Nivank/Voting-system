Voting System - README Documentation

About

The Voting System is a Python-based application designed to manage and execute digital voting
securely. It supports adding candidates, casting votes using face recognition, and storing results in
CSV format. Built with Flask for the web interface and OpenCV for image processing.

Features

1 Add and manage candidates with face recognition. 

2 Secure voting interface via Flask web app.

3 Vote data stored in CSV files for easy retrieval.

4 Lightweight and easy to deploy locally.

5 Supports automatic browser launch via batch file.

Installation & Setup

1 Ensure Python 3.11 and pip are installed.

2 Clone the repository: git clone https://github.com/Nivank/Voting-system.git

3 Navigate to the project folder: cd Voting-system

4 Create and activate a virtual environment: python -m venv .venv && .venv\Scripts\activate

5 Install dependencies: pip install -r requirements.txt

Usage

1 Run add_faces.py to register candidates using their images.

2 Start the server with: python server.py

3 Access the system via browser at http://127.0.0.1:5000

4 Votes are saved in Votes.csv automatically.

Contact

GitHub: https://github.com/Nivank 
Email: nivankn@gmail.com.com

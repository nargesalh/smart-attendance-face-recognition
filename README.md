# Face Recognition Attendance System

A smart attendance system based on face recognition, designed and implemented using **Clean Architecture** principles.

## 📌 Features
- Face recognition–based attendance registration
- Clean and modular architecture (Core, Infra, Presentation)
- SQLite database for data persistence
- InsightFace engine for face recognition
- Simple graphical user interface using Tkinter

  

## 🧱 Project Structure
.
├── core
│   ├── entities.py
│   ├── ports.py
│   └── exceptions.py
├── infra
│   ├── persistence
│   │   └── sqlite_repository.py
│   ├── recognition
│   │   ├── face_index.py
│   │   └── insightface_engine.py
│   └── utils
├── presentation
│   └── ui_tk
├── compat
│   └── backend_facade.py
├── run_ui.py
└── requirements.txt


## 🧠 Architecture
This project follows **Clean Architecture**, separating responsibilities into different layers:

- **Core**: Business logic, entities, and interfaces
- **Infrastructure**: Database access and face recognition implementation
- **Presentation**: User interface layer
- **Compat**: Facade layer to simplify communication between UI and backend

This design improves maintainability, scalability, and testability.

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
```


Install dependencies:
```
pip install -r requirements.txt
```


Run the application:
```
python run_ui.py
```

Technologies Used:
Python
InsightFace
SQLite
Tkinter

🎓 Purpose
This project was developed as a final academic project to demonstrate the practical application of:
Clean Code principles
Clean Architecture
Face recognition in real-world systems

👤 Author
Narges Aliheydari

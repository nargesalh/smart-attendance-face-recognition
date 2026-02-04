# Attendance Clean Architecture

This is a cleaned, layered re-structure of your face attendance project:
- **core/** — domain entities and ports (interfaces)
- **infra/** — infrastructure adapters (SQLite, InsightFace, utils, config)
- **usecases/** — application logic (attendance runtime)
- **presentation/** — UI (Tkinter), decoupled via a compatibility facade
- **compat/** — thin facade that keeps the old surface area for easy migration

## How to run

1) Install dependencies (Python 3.10+ recommended):
```
pip install -r requirements.txt
```

2) (Optional) Set environment variables to customize paths:
- `ATT_DATA_DIR` (default: `./data`)
- `ATT_IMAGES_DIR` (default: `./data/face_images`)
- `ATT_DB_PATH` (default: `./data/attendance.db`)
- `INSIGHTFACE_ROOT` (default: `~/.insightface`)

3) Launch the Tkinter app:
```
python -m attendance_clean.presentation.ui_tk.app
```
or
```
python run_ui.py
```

## Notes

- The original DB schema, recognition flow (tracking + voting), and UI behaviors are preserved, but
  the **logic is now separated** into layers so you can unit test and swap implementations more easily.
- The compatibility layer exposes the same names (`DB`, `FaceEngine`, `FaceIndex`, `AttendanceRuntime`)
  used in your old UI, so migration is straightforward.

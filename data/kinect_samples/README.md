# Kinect Sample Data

This folder is intended for sample RGB-D frames captured with the Microsoft Kinect Xbox 360.

Due to file size constraints, raw sensor frames are not included in this repository.

## How to generate your own samples

Run the system with your Kinect connected:

```bash
python python_scripts/main.py
```

Output frames and logs will be saved automatically to the `logs/` and `Output/` directories.

## Sample log data

Pre-recorded behavioral log files (CSV and JSON) are available in the `logs/` directory,
captured during the single-subject validation protocol described in the paper.

# Travel Time Prediction

Team coursework for Introduction to Machine Learning at Beijing Institute of Technology, Spring 2026. This repository contains an AI-assisted portfolio revision of the coursework.

## What it does

The model estimates how long a vehicle takes to travel along a road segment between 08:00 and 09:00, in two-minute intervals. It uses historical traffic records and same-day observations from 06:00 to 08:00. LightGBM learns from the road ID, time of day, weekday, and early-morning traffic summaries.

## Run on Windows

If `Data.rar` is in your Downloads folder and the dependencies are installed, open `predict.py` in VS Code and click Run. The script will find the archive and save the `results` folder beside `predict.py`.

Use Python 3.12 and the course-provided `Data.rar` archive. Open PowerShell in the project folder and run:

```powershell
python -m pip install -r requirements.txt
python predict.py "$env:USERPROFILE\Downloads\Data.rar"
```

The archive command uses Windows `tar`. On a system where `tar` cannot read RAR files, extract `gy_link_travel_time_part3.txt` with an archive utility and pass the extracted text file to `predict.py` instead. Replace the example path with your own file location. The course dataset is distributed separately.

The script writes `results/validation.txt` and `results/june_forecast.csv`. The CSV averages daily predictions over the 30 June days, giving one forecast per road segment and two-minute interval.

## Verified local run

| Measurement | Result |
| --- | ---: |
| Training records | 284,738 |
| Held-out records, May 18–31, 2017 | 51,902 |
| Historical-average baseline error (MAPE) | 59.49% |
| LightGBM error (MAPE) | 36.38% |
| June forecasts | 3,960 across 132 road segments |

MAPE is the average absolute percentage error. The baseline uses the training average for each road and time slot. The evaluation uses later dates held out from training, with features drawn only from information available before 08:00 on each forecast day. The final June predictions use a model refit on the available pre-June history. The scores above measure performance on historical May observations. June ground-truth accuracy was not measured.

## Credit

The original coursework was a three-person team project. The portfolio revision was prepared with AI assistance and uses a revised evaluation and forecasting script. The original team submission and this later revision are separate versions.

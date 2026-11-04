# SwiftDrop Lagos: Delivery Dispatch (Group 5)

This is the repo submission for group 5 for DAT613 CA. PAU MSc Data Science 5. DAT613 Programming for Data Science.

## Question
How many motorbike riders does each SwiftDrop hub (Ikeja, Yaba, Lekki) need?

## About
This repository turns a flat stream of hourly orders into a rider plan, using NumPy and Matplotlib only.

## How to run

1. Install the two libraries:
   ```
   pip install -r requirements.txt
   ```
2. Put `group05_logistics.npz` in the `data/` folder. The data file is not in this repository.
3. Run the pipeline from the repository folder:
   ```
   python3 pipeline.py data/group05_logistics.npz
   ```
4. Run the tests (they build changed copies of the data in a temporary folder):
   ```
   python3 test_pipeline.py
   ```
5. Optional speed test, Python loops against NumPy:
   ```
   python3 benchmark.py data/group05_logistics.npz
   ```

The pipeline works on any file of this kind, including a new file with the same problems in different places.

## Output

The answers and the evidence for each data problem print to the screen. Three charts are saved in `figures/`:

- `riders_per_hub.png`: riders each hub needs
- `riders_by_hour.png`: average riders per hub by hour, with orders as faint bars
- `longest_trip_per_area.png`: longest one-way trip per area against the 60 minute target

## Answers

- Ikeja needs 8 riders, Yaba 24 and Lekki 9 (41 in total), enough to cover 95% of open hours.
- The busiest hour is 12:00 by orders and 18:00 by rider workload; the longest trip is Festac at 18:00 (58.9 min), so every area is within 60 minutes.

## Pipeline and shapes

| Stage | File | Shape out |
|---|---|---|
| Load the raw order stream | `src/structure.py` | `(N,)` |
| Reshape to day × hour × area | `src/structure.py` | `(days, 14, 12)` |
| Find and fix the three data problems | `src/problems.py` | `(days, 14, 12)`, `(12, 2)`, `(14,)` |
| Distance from every area to every hub | `src/routing.py` | `(12, 3)` |
| Nearest hub and one-way trip time | `src/routing.py` | `(12,)`, `(14, 12)` |
| Rider-minutes and riders per hub per hour | `src/riders.py` | `(days, 14, 3)` |
| Riders per hub (95th percentile), charts | `src/answers.py` | `(3,)` |
| Everything in one call | `pipeline.py` | results dict |

## Who owns what

| Part | File | Owner |
|---|---|---|
| Data & structure | `src/structure.py` | Ayomide Oluwadiya |
| Data problems | `src/problems.py` | Victor Olanrewaju |
| Analysis: routing | `src/routing.py` | Oluwasegun Oni |
| Analysis: riders | `src/riders.py` | James Ofor |
| Business answers & charts | `src/answers.py` | Akpovoke Okwowe |
| Integration, README, benchmark, report | `pipeline.py`, `benchmark.py` | Chukwudi Ofoma |

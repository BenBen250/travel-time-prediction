"""Forecast morning travel time from the BIT coursework dataset."""
import argparse
from pathlib import Path
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd


def read_data(path):
    parts = []
    for chunk in pd.read_csv(
        path, sep=";", dtype={"link_ID": str}, chunksize=400_000
    ):
        start = chunk["time_interval"].str.slice(12, 17)
        morning = start.ge("06:00") & start.lt("09:00")
        selected = chunk.loc[morning, ["link_ID", "date", "travel_time"]].copy()
        selected["slot"] = start[morning].str.slice(0, 2).astype(int) * 60
        selected["slot"] += start[morning].str.slice(3, 5).astype(int)
        parts.append(selected)
    data = pd.concat(parts, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    return data.dropna().loc[lambda rows: rows["travel_time"] > 0]


def features(rows, early):
    rows = rows.merge(early, on=["link_ID", "date"], how="left")
    rows["weekday"] = rows["date"].dt.dayofweek
    return rows


def fit_predict(train, target):
    columns = ["link_ID", "slot", "weekday", "early_mean", "early_last"]
    link_type = pd.CategoricalDtype(sorted(train["link_ID"].unique()))
    x_train, x_target = train[columns].copy(), target[columns].copy()
    for frame in (x_train, x_target):
        frame["link_ID"] = frame["link_ID"].astype(link_type)
    model = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.06, num_leaves=31,
        random_state=42, n_jobs=4, verbosity=-1,
    )
    model.fit(x_train, np.log1p(train["travel_time"]))
    return np.maximum(np.expm1(model.predict(x_target)), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", type=Path, nargs="?",
        help="Extracted traffic text file or Data.rar (default: ~/Downloads/Data.rar)",
    )
    parser.add_argument("--output", type=Path, help="Folder for forecasts and validation results")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "results" if args.data is None else Path("results")
    if args.data is None:
        args.data = Path.home() / "Downloads" / "Data.rar"
    if not args.data.is_file():
        parser.error(f"Traffic data file not found: {args.data}. Pass the path to Data.rar or the extracted text file.")
    if args.data.suffix.lower() == ".rar":
        with subprocess.Popen(
            ["tar", "-xOf", str(args.data), "gy_link_travel_time_part3.txt"],
            stdout=subprocess.PIPE,
        ) as archive:
            try:
                data = read_data(archive.stdout)
            except BaseException:
                archive.kill()
                raise
            if archive.wait() != 0:
                raise RuntimeError("Could not read the traffic archive.")
    else:
        data = read_data(args.data)

    # These measurements are available before the forecast starts at 08:00.
    early = (data[data["slot"] < 480].sort_values("slot")
             .groupby(["link_ID", "date"])["travel_time"]
             .agg(early_mean="mean", early_last="last").reset_index())
    june = pd.Timestamp("2017-06-01")
    history = data[(data["date"] < june) & (data["slot"] >= 480)]
    history = features(history, early)
    cutoff = pd.Timestamp("2017-05-18")
    train = history[history["date"] < cutoff]
    valid = history[history["date"] >= cutoff]
    june_days = pd.DatetimeIndex(
        sorted(early.loc[early["date"].between(june, "2017-06-30"), "date"].unique())
    )
    if train.empty or valid.empty or june_days.empty:
        raise ValueError("The dataset must contain older morning data, late May targets, and June context.")

    actual = valid["travel_time"].to_numpy()
    means = train.groupby(["link_ID", "slot"])["travel_time"].mean()
    key = pd.MultiIndex.from_frame(valid[["link_ID", "slot"]])
    baseline = means.reindex(key).fillna(train["travel_time"].mean()).to_numpy()
    predicted = fit_predict(train, valid)
    baseline_mape = 100 * np.mean(np.abs(baseline - actual) / actual)
    model_mape = 100 * np.mean(np.abs(predicted - actual) / actual)

    links = sorted(history["link_ID"].unique())
    grid = pd.MultiIndex.from_product(
        [links, june_days, range(480, 540, 2)], names=["link_ID", "date", "slot"]
    ).to_frame(index=False)
    grid["predicted_travel_time_seconds"] = fit_predict(history, features(grid, early))
    forecast = (grid.groupby(["link_ID", "slot"], as_index=False)
                ["predicted_travel_time_seconds"].mean())
    forecast["interval_start"] = forecast["slot"].map(
        lambda minute: f"{minute // 60:02d}:{minute % 60:02d}"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    forecast[["link_ID", "interval_start", "predicted_travel_time_seconds"]].to_csv(
        args.output / "june_forecast.csv", index=False, float_format="%.3f"
    )
    report = (
        f"Training rows: {len(train):,}\n"
        f"Validation dates: {valid['date'].min():%Y-%m-%d} through {valid['date'].max():%Y-%m-%d}\n"
        f"Validation rows: {len(valid):,}\n"
        f"Historical-mean baseline MAPE: {baseline_mape:.2f}%\n"
        f"LightGBM MAPE: {model_mape:.2f}%\n"
        f"June dates with early records: {len(june_days)}\n"
        f"Road segments: {len(links)}\n"
        f"June forecast rows: {len(forecast):,}\n"
    )
    (args.output / "validation.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

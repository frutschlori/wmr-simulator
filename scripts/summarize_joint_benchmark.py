import argparse

from wmr_simulator.joint_tuning.benchmark import comparison_table, load_runs, summarize


def main():
    parser = argparse.ArgumentParser(description="Summarize joint-tuning benchmark records.")
    parser.add_argument("--dir", type=str, default="results/joint_tuning_benchmark")
    parser.add_argument("--stage", type=str, default=None, help="Only records from this stage.")
    parser.add_argument("--plot", type=str, default=None,
                        help="Write a cross-run comparison figure to this path, with one "
                             "group per distinct value of --group-by.")
    parser.add_argument("--group-by", type=str, default="mode")
    parser.add_argument("--columns", type=str, default=None,
                        help="Comma-separated comparison_row fields; prints a table instead of "
                             "one line per record.")
    args = parser.parse_args()

    records = sorted(
        (
            record
            for record in load_runs(args.dir)
            if args.stage is None or record["meta"]["stage"] == args.stage
        ),
        key=lambda entry: entry["meta"]["timestamp"],
    )
    if args.plot:
        from wmr_simulator.visualization.joint_tuning import plot_benchmark_comparison

        plot_benchmark_comparison(
            records,
            group_by=args.group_by,
            out_path=args.plot,
            title=f"Joint tuning benchmark — {args.stage or 'all stages'}",
        )
    if args.columns:
        print(comparison_table(records, tuple(args.columns.split(","))))
    else:
        for record in records:
            print(summarize(record))
    print(f"{len(records)} record(s) in {args.dir}")


if __name__ == "__main__":
    main()

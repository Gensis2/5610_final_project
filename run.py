import cnn_run
import snn_run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN or SNN model")
    parser.add_argument("--model", type=str, choices=["cnn", "snn"], required=True, help="Model type to run")
    parser.add_argument("--case_study", action="store_true", help="Run case study")

    args = parser.parse_args()

    print(args.case_study)

    if args.case_study:
        print("Running case study...")
    else:
        print("Running single run...")

    if args.model == "cnn" and not args.case_study:
        cnn_run.cnn_run()
    elif args.model == "snn" and not args.case_study:
        snn_run.single_run()
    elif args.model == "cnn" and args.case_study:
        cnn_run.case_study()
    elif args.model == "snn" and args.case_study:
        snn_run.case_study()
    else:
        print("Invalid arguments. Please specify a valid model and options.")
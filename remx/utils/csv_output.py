from typing import List


def csv_result(pred_output: List[dict], file_name: str):
    import csv

    file_name += ".csv"

    # Define the file name and column names
    fieldnames = ["image location", "coordinate"]

    # Open the CSV file for writing
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write data from pred_output
        for value in pred_output:
            # Write a row with image location and coordinate
            writer.writerow(
                {
                    "image location": value["img_loc"],
                    "coordinate": value["max_confidence_coordinate"],
                }
            )

    print(f'CSV file "{file_name}" created successfully.')

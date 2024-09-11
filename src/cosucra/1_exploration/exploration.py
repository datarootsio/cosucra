import pandas as pd
import matplotlib.pyplot as plt
import os


def group_tags_by_process(file_path, sheet_name="Feuil1"):
    """
    This function reads an Excel file and returns a dictionary where each process is mapped to an array of associated tags.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet containing the data (default: 'Feuil1').

    Returns:
    - dict: A dictionary with processes as keys and a list of associated tags as values.
    """
    # Load the Excel sheet into a pandas DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Ensure the first two columns are "Process" and "Tags"
    if "Process" not in df.columns or "Tags" not in df.columns:
        raise ValueError(
            "Columns 'Process' and 'Tags' must be present in the Excel file."
        )

    # Group tags by process and convert to a dictionary
    grouped = df.groupby("Process")["Tags"].apply(list).to_dict()

    # Create the process_sheets variable to store the tags by process
    process_sheets = grouped

    return process_sheets


def load_data(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_excel(file_path, encoding="latin-1")


def convert_second_column_to_float(data_dict):
    """
    Converts the second column of each DataFrame in the dictionary to float.

    Parameters:
    - data_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.

    Returns:
    - dict: The updated dictionary with the second column of each DataFrame converted to float.
    """
    for sheet_name, df in data_dict.items():
        if df.shape[1] >= 2:  # Ensure there are at least two columns
            second_column_name = df.columns[1]
            df[second_column_name] = pd.to_numeric(
                df[second_column_name], errors="coerce"
            )
    return data_dict


def convert_object_columns_to_float(data_dict):
    """
    Converts all object columns in each DataFrame in the dictionary to float.

    Parameters:
    - data_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.

    Returns:
    - dict: The updated dictionary with all object columns converted to float.
    """
    for sheet_name, df in data_dict.items():
        # Iterate over each column
        for col in df.columns:
            if df[col].dtype == "object":
                # Attempt to convert object column to float
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return data_dict


def count_nulls_in_sheets(data_dict):
    """
    Counts the number of null values in each column of each DataFrame in the dictionary.

    Parameters:
    - data_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.

    Returns:
    - dict: A dictionary where keys are sheet names and values are Series with column names as index and count of null values as values.
    """
    null_counts = {}
    for sheet_name, df in data_dict.items():
        null_counts[sheet_name] = df.isna().sum()
    return null_counts


def remove_nulls_from_dict(data_dict, subset=None):
    """
    Removes rows with null values from each DataFrame in the dictionary.

    Parameters:
    - data_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.
    - subset (list, optional): List of column names to consider for identifying null values.
                                If None, considers all columns.

    Returns:
    - dict: The dictionary with each DataFrame having rows containing null values removed.
    """
    cleaned_data_dict = {}

    for sheet_name, df in data_dict.items():
        # Apply the remove_nulls function to each DataFrame
        cleaned_df = remove_nulls(df, subset)
        cleaned_data_dict[sheet_name] = cleaned_df

    return cleaned_data_dict


# Define the remove_nulls function used above
def remove_nulls(df, subset=None):
    """
    Removes rows with null values from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - subset (list, optional): List of column names to consider for identifying null values.
                                If None, considers all columns.

    Returns:
    - pd.DataFrame: The DataFrame with rows containing null values removed.
    """
    if subset:
        df_cleaned = df.dropna(subset=subset)
    else:
        df_cleaned = df.dropna()

    return df_cleaned


def detect_outliers(df):
    """
    Detects outliers in a DataFrame using the IQR method.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process. It should have numeric columns.

    Returns:
    - pd.DataFrame: A DataFrame indicating outliers with a boolean flag.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each numeric column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define outliers as values outside 1.5*IQR range
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    return outliers


def summarize_sheets_with_outlier_counts(data_dict):
    """
    Summarizes each DataFrame in the dictionary and counts the outliers.

    Parameters:
    - data_dict (dict): Dictionary with sheet names as keys and DataFrames as values.

    Returns:
    - pd.DataFrame: A DataFrame summarizing outlier counts for each sheet.
    """
    outlier_counts = []

    for sheet_name, df in data_dict.items():
        # Apply the detect_outliers function to each DataFrame
        outliers_df = detect_outliers(df.select_dtypes(include="number"))

        # Count the number of outliers in each column
        outlier_count_per_column = outliers_df.sum()
        total_outliers = outlier_count_per_column.sum()

        # Collect the results
        outlier_counts.append(
            {
                "Sheet Name": sheet_name,
                "Total Outliers": total_outliers,
                **outlier_count_per_column.to_dict(),  # Include column-wise outlier counts
            }
        )

    # Create a DataFrame from the outlier counts
    outlier_counts_df = pd.DataFrame(outlier_counts)

    return outlier_counts_df


def summarize_sheets(data_dict):
    """
    Summarizes the data in each DataFrame within the provided dictionary.

    Parameters:
    - data_dict (dict): A dictionary where keys are sheet names and values are pandas DataFrames.

    Returns:
    - pd.DataFrame: A DataFrame containing summary statistics for each sheet.
    """
    summary_list = []

    for sheet_name, df in data_dict.items():
        # Generate descriptive statistics for numeric columns
        df = pd.DataFrame(df)
        summary_df = df.describe(include="all")
        summary_list.append(summary_df)

    return summary_list


def plot_sheets_data(sheet_dict):
    """
    This function takes a dictionary where the keys are sheet names and the values are DataFrames.
    Each DataFrame is expected to have two columns: the first column is a date, and the second column is float values.
    It will generate a line plot for each sheet.

    Parameters:
    sheet_dict (dict): Dictionary with sheet names as keys and DataFrames as values.
    """

    for sheet_name, df in sheet_dict.items():
        # Convert the first column to datetime if it's not already
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

        # Set up the plot
        plt.figure(figsize=(10, 6))

        # Plot the date (1st column) vs float values (2nd column)
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f"{sheet_name}")

        # Add titles and labels
        plt.title(f"Data from {sheet_name}")
        plt.xlabel("Date")
        plt.ylabel("Float Values")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display legend
        plt.legend()

        # Adjust layout for better appearance
        plt.tight_layout()

        # Show the plot
        plt.show()


def custom_describe(df):
    # Exclude Timestamp columns
    non_timestamp_cols = df.select_dtypes(exclude=["datetime64"]).columns

    # Calculate statistics for the non-Timestamp columns
    stats = {}
    for col in non_timestamp_cols:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "median": df[col].median(),
            "mean": df[col].mean(),
        }

    # Convert the statistics dictionary to a DataFrame
    stats_df = pd.DataFrame(stats).T  # Transpose to have columns as rows

    return stats_df


def process_and_summarize_data(data_dict, process_name, null_threshold=0.1):
    """
    Processes each DataFrame in the dictionary: checks null values, cleans data, and returns detailed results in a DataFrame.
    Saves plots to a specific directory structure.

    Parameters:
    - data_dict (dict): Dictionary with sheet names as keys and DataFrames as values.
    - process_name (str): Name of the process used to create directory for saving plots.
    - null_threshold (float): The threshold for deciding whether to drop or fill null values (default is 0.1 for 10%).

    Returns:
    - pd.DataFrame: A DataFrame containing the results for each sheet.
    """
    results_list = []

    # type conversion
    data_dict = convert_object_columns_to_float(data_dict)
    # Ensure the working directory is correct (if necessary)
    output_dir = f"plots/{process_name}"
    os.makedirs(output_dir, exist_ok=True)

    # convert object type to float type col

    # Process each DataFrame
    for sheet_name, df in data_dict.items():
        # Count the number of null values before cleaning
        non_timestamp_cols = df.select_dtypes(exclude=["datetime64"]).columns

        # Count the number of null values before cleaning for non-Timestamp columns
        null_count_before = df[non_timestamp_cols].isna().sum()

        # Check if null percentage exceeds the threshold
        null_percentage = null_count_before / len(df)
        if (null_percentage > null_threshold).any():
            # Fill null values with median
            # cleaned_df = df.fillna(df.median(numeric_only=True))
            cleaning_method = "filled with median"
        else:
            # Drop rows with null values
            # cleaned_df = df.dropna()
            cleaning_method = "dropped rows with null values"

        # Generate and save plot
        plt.figure(figsize=(10, 6))
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f"{sheet_name}")
        plt.title(f"Data from {sheet_name}")
        plt.xlabel("Date")
        plt.ylabel("Float Values")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{sheet_name}_plot.png")
        plt.savefig(plot_filename)
        plt.close()

        # Summarize outliers
        outliers_df = detect_outliers(df.select_dtypes(include="number"))
        outlier_count_per_column = outliers_df.sum()
        total_outliers = outlier_count_per_column.sum()

        # Descriptive

        # Collect results for this sheet
        result = {
            "Sheet Name": sheet_name,
            "Cleaning Method": cleaning_method,
            "Null Counts Before CLEANING": null_count_before,
            "Plot Filename": plot_filename,
            "Total Outliers": total_outliers,
            "Summary Statistics": custom_describe(df).to_dict(),
            "Outlier Counts": outlier_count_per_column.to_dict(),
        }

        results_list.append(result)

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Normalize nested columns manually
    # Expand 'Summary Statistics' and 'Outlier Counts' into separate columns
    summary_stats_df = pd.json_normalize(results_df["Summary Statistics"])
    outlier_counts_df = pd.json_normalize(results_df["Outlier Counts"])

    # Concatenate everything into a final DataFrame
    final_results_df = pd.concat(
        [
            results_df.drop(["Summary Statistics", "Outlier Counts"], axis=1),
            summary_stats_df,
            outlier_counts_df,
        ],
        axis=1,
    )

    # Print the DataFrame
    print(final_results_df)

    return final_results_df

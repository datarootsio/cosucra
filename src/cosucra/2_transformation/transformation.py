import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_df_as_png(df, filename="output.png", num_rows=20):
    """
    Save the first `num_rows` rows of a DataFrame as a PNG image.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    filename (str): The name of the output PNG file.
    num_rows (int): The number of rows to display in the image.
    """

    # Select the first `num_rows` rows
    df_selected = df.head(num_rows)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the size as needed

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Render the DataFrame as a table in the axis
    table = ax.table(
        cellText=df_selected.values,
        colLabels=df_selected.columns,
        cellLoc="center",
        loc="center",
    )

    # Adjust the table's size and scale
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the figure as a PNG
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def aggregate_and_merge_by_hour(data_dict):
    """
    Aggregates timestamp data by hours and merges the DataFrames from the provided dictionary.

    Parameters:
    data_dict (dict): Dictionary where keys are the names for the columns and values are DataFrames
                      containing 'TimeStamp' and one data column.

    Returns:
    pd.DataFrame: A merged DataFrame aggregated by hour with columns named by the dictionary keys.
    """

    # Initialize an empty list to store resampled DataFrames
    resampled_dfs = []

    # Loop through the dictionary and process each DataFrame
    for key, df in data_dict.items():
        # Check that the DataFrame has 2 columns (TimeStamp and Values)
        if df.shape[1] == 2:
            # Rename the second column (values) to the key name
            df.columns = ["TimeStamp", key]
        else:
            raise ValueError(
                f"DataFrame associated with key {key} doesn't have exactly 2 columns."
            )

        # Convert the 'TimeStamp' column to datetime format
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

        # Convert the second column to numeric, forcing errors to NaN
        df[key] = pd.to_numeric(df[key], errors="coerce")

        # Set 'TimeStamp' as the index
        df.set_index("TimeStamp", inplace=True)

        # Resample the data by hour and aggregate (mean)
        df_resampled = df.resample("30T").mean()

        df_resampled = df_resampled.round(2)
        # Append the resampled DataFrame to the list
        resampled_dfs.append(df_resampled)

    # Merge all resampled DataFrames on the index (TimeStamp)
    merged_df = pd.concat(resampled_dfs, axis=1)

    # Reset the index to bring 'TimeStamp' back as a column
    merged_df.reset_index(inplace=True)

    return merged_df


def apply_transformation(process_array, data_dict):
    results = {}

    for process_name in process_array:
        # Get the DataFrame for the current process from the dictionary
        process_data_df = data_dict.get(process_name)

        # Apply the aggregation and merging function
        if process_data_df is not None:
            # Assuming transformation.aggregate_and_merge_by_hour is the function you want to apply
            result_df = aggregate_and_merge_by_hour(process_data_df)
            results[process_name] = result_df
        else:
            print(f"Data for {process_name} not found.")

    return results


import pandas as pd


def shift_non_timestamp_columns(df, shift_value=1):
    """
    Applies a shift to all columns except the 'TimeStamp' column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing a 'TimeStamp' column and other columns to shift.
    shift_value (int): The number of rows to shift (default is 1).

    Returns:
    pd.DataFrame: A DataFrame with shifted values for all non-'TimeStamp' columns.
    """

    # Ensure 'TimeStamp' column is not modified
    timestamp_col = df["TimeStamp"]

    # Shift all columns except 'TimeStamp' by the specified shift_value
    shifted_df = df.drop(columns=["TimeStamp"]).shift(shift_value)

    # Add back the 'TimeStamp' column
    shifted_df["TimeStamp"] = timestamp_col

    # Reorder columns to place 'TimeStamp' first
    shifted_df = shifted_df[
        ["TimeStamp"] + [col for col in shifted_df.columns if col != "TimeStamp"]
    ]

    return shifted_df


def shift_custom_columns_generic(df, shifts, boundaries):
    """
    Applies different shifts to specific ranges of columns in a DataFrame based on boundaries.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'TimeStamp' and other columns to shift.
    shifts (dict): Dictionary where keys are boundary names and values are shift values.
                   For example: {'A': 1, 'B': 2, 'C': 3}
    boundaries (list): List of boundary column names in the order they appear.
                        For example: ['A', 'B', 'C']

    Returns:
    pd.DataFrame: A DataFrame with shifted values based on the provided ranges.
    """

    # Ensure 'TimeStamp' column is not modified
    timestamp_col = df["TimeStamp"]

    # Get column names
    columns = df.columns.tolist()

    # Identify column indices for boundaries
    boundary_indices = {boundary: columns.index(boundary) for boundary in boundaries}

    # Add the index of the last boundary plus one for range handling
    boundary_indices["end"] = len(columns)

    # Initialize list to hold the shifted DataFrames
    shifted_parts = []

    # Shift columns based on boundaries
    prev_index = 1  # Start right after 'TimeStamp'
    for i, boundary in enumerate(boundaries):
        current_index = boundary_indices[boundary]
        shift_value = shifts.get(boundary, 0)

        # Extract and shift the relevant columns
        if i == len(boundaries) - 1:
            # If it's the last boundary, shift until the end of the columns
            part = df.iloc[:, prev_index : boundary_indices["end"]].shift(shift_value)
        else:
            # Shift columns between the current and next boundary
            next_boundary = boundaries[i + 1]
            next_index = boundary_indices[next_boundary]
            part = df.iloc[:, prev_index:next_index].shift(shift_value)

        # Append the shifted part to the list
        shifted_parts.append(part)

        # Update prev_index for the next range
        prev_index = current_index

    # Combine all shifted parts
    shifted_df = pd.concat(shifted_parts, axis=1)

    # Add back the 'TimeStamp' column
    shifted_df["TimeStamp"] = timestamp_col

    # Reorder columns to place 'TimeStamp' first
    shifted_df = shifted_df[
        ["TimeStamp"] + [col for col in df.columns if col != "TimeStamp"]
    ]

    return shifted_df


def window_by_column_name(df, start_col_name, num_cols=2):
    """
    Extracts a window of columns from the DataFrame starting from a specific column name.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to extract columns.
    start_col_name (str): The name of the starting column for the window.
    num_cols (int): The number of columns to include in the window (default is 4).

    Returns:
    pd.DataFrame: A subset of the DataFrame with the specified columns.
    """

    # Ensure the starting column name exists in the DataFrame
    if start_col_name not in df.columns:
        raise ValueError(f"Column '{start_col_name}' not found in DataFrame.")

    # Get the list of column names
    columns = df.columns.tolist()

    # Find the starting index for the given column name
    start_col_index = columns.index(start_col_name)

    # Determine the end index for the column window
    end_col_index = start_col_index + num_cols

    # Slice the column names based on the specified indices
    selected_columns = columns[start_col_index:end_col_index]

    # Ensure we do not exceed the number of columns in the DataFrame
    if len(selected_columns) < num_cols:
        print(
            f"Warning: Only {len(selected_columns)} columns available from '{start_col_name}'."
        )

    # Select these columns from the DataFrame
    window_df = df[selected_columns]

    return remove_duplicate_columns(window_df)


def remove_duplicate_columns(df):
    """
    Removes duplicate columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove duplicate columns.

    Returns:
    pd.DataFrame: A DataFrame with duplicate columns removed.
    """

    # Transpose the DataFrame to work with columns as rows
    df_T = df.T

    # Identify duplicate columns
    duplicated_columns = df_T.duplicated(keep="first")

    # Filter out the duplicate columns
    df_unique = df_T[~duplicated_columns].T

    return df_unique


def get_columns_data(df, column_names):
    """
    Extracts data from specific columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to extract columns.
    column_names (list): List of column names to extract.

    Returns:
    pd.DataFrame: A DataFrame containing only the specified columns.
    """
    # Ensure the column names are in the DataFrame
    valid_columns = [col for col in column_names if col in df.columns]

    # Extract the specified columns
    return df[valid_columns]


def merge_datasets_on_timestamp(dfs):
    """
    Merges a list of DataFrames on the 'TimeStamp' column.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames to merge.

    Returns:
    pd.DataFrame: A single DataFrame merged on the 'TimeStamp' column.
    """

    if not dfs:
        raise ValueError("No DataFrames provided for merging.")

    # Start with the first DataFrame
    merged_df = dfs[0]

    # Iteratively merge with the remaining DataFrames
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="TimeStamp", how="outer")

    return merged_df


def perform_pca(df, n_components=2):
    """
    Perform PCA on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with a 'TimeStamp' column and other numerical columns.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: A DataFrame with the principal components.
    """

    # Step 1: Handle missing values (e.g., fill with column mean)
    df_filled = df.fillna(df.mean())

    # Step 2: Standardize the data (excluding the 'TimeStamp' column)
    features = df_filled.drop(columns=["TimeStamp"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features, y=[df_filled["FT_T2_Pois_Entier"], df_filled["FT_G2_0402_MS"]]
    )

    # Step 3: Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(
        scaled_features, y=[df_filled["FT_T2_Pois_Entier"], df_filled["FT_G2_0402_MS"]]
    )

    # Create a DataFrame with the principal components
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

    # Step 4: Add the 'TimeStamp' column back to the PCA result
    pca_df = pd.concat(
        [df_filled[["TimeStamp"]].reset_index(drop=True), pca_df], axis=1
    )

    return pca_df, pca


def get_top_values_indices(array, top_n=5):
    """
    Finds the top N values in an array and returns their indices.

    Parameters:
    array (np.ndarray): The array to find top values from.
    top_n (int): The number of top values to find (default is 5).

    Returns:
    tuple: A tuple containing:
        - The top N values.
        - Their corresponding indices in the array.
    """
    # Convert the array to a numpy array if it's not already
    array = np.array(array)

    # Get the indices of the top N values
    top_indices = np.argsort(array)[-top_n:][::-1]  # Sort in descending order

    # Get the top N values
    top_values = array[top_indices]

    return top_values, top_indices


def get_column_names_from_indices(df, indices):
    """
    Gets column names from the DataFrame based on the given indices.

    Parameters:
    df (pd.DataFrame): The DataFrame with column names to retrieve.
    indices (list or np.ndarray): The indices of the columns to retrieve names for.

    Returns:
    list: A list of column names corresponding to the given indices.
    """
    return df.columns[indices].tolist()


def plot_explained_variance(df):
    """
    Performs PCA on the DataFrame and plots the explained variance ratio.

    Parameters:
    df (pd.DataFrame): The DataFrame to perform PCA on.

    Returns:
    None
    """

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # Plot the explained variance ratio (scree plot)
    plt.figure(figsize=(12, 6))

    # Scree plot
    plt.subplot(1, 2, 1)
    sns.barplot(
        x=list(range(1, len(explained_variance_ratio) + 1)), y=explained_variance_ratio
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")

    # Cumulative explained variance plot
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(cumulative_explained_variance) + 1),
        cumulative_explained_variance,
        marker="o",
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance Plot")

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    """
    Calculates and plots the correlation matrix of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to calculate the correlation matrix.

    Returns:
    None
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix")
    plt.show()

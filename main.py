import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.time_series import setup, compare_models, pull, save_model, load_model, tune_model, plot_model

# Load dataset if exists
if os.path.exists("dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)
    if 'df' not in st.session_state:
        st.session_state['df'] = df

# Sidebar setup
with st.sidebar:
    st.image("https://www.politico.com/interactives/uploads/image-service/2023/3/15/e516e3bf6e-420.png", width=150)
    st.title("Trump Will Tell")
    choice = st.radio("Select the task", ["Upload", "Filter", "Profiling", "Target", "Comparison", "Final Plot"])

# Upload section
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, index_col=None)
            st.session_state['df'] = df  # Save dataset to session_state
            st.success("Dataset uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Filter section
if choice == "Filter":
    if 'df' in st.session_state:
        st.title("Filter Your Dataset")

        # Select column to filter
        df = st.session_state['df']
        column = st.selectbox("Select a column to filter", options=df.columns)

        # Select values in the column to filter
        unique_values = df[column].unique()
        selected_values = st.multiselect(f"Select values for {column}", options=unique_values)

        # Filter dataset based on selections
        if selected_values:
            filtered_df = df[df[column].isin(selected_values)].copy()
            st.session_state['filtered_df'] = filtered_df  # Save filtered dataset to session_state
            st.success(f"Filtered dataset with {column} in {selected_values}")
        else:
            st.session_state['filtered_df'] = df  # Default to entire dataset
            st.warning("No values selected. Using the entire dataset.")

        # Display filtered dataset
        st.dataframe(st.session_state['filtered_df'])
    else:
        st.error("No dataset found. Please upload a dataset first.")

# Profiling section
if choice == "Profiling":
    if 'filtered_df' in st.session_state:
        st.title("Exploratory Data Analysis")

        profiling_data = st.session_state['filtered_df']
        if not profiling_data.empty:
            st.info("Profiling is based on the filtered dataset.")
            profile = ydata_profiling.ProfileReport(profiling_data, explorative=True)
            st_profile_report(profile)
        else:
            st.error("Filtered dataset is empty. Please adjust your filter or upload a valid dataset.")
    else:
        st.error("No filtered dataset found. Please apply a filter first.")

# Target section
if choice == "Target":
    if 'filtered_df' in st.session_state:
        st.title("Select Target and Date Column")

        # Use filtered dataset
        df = st.session_state['filtered_df']
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Predefined date column (update the name as per your dataset)
        predefined_date_column = st.selectbox("Select the date column", options=df.columns, help="Choose the column to forecast.")
        if predefined_date_column not in df.columns:
            st.error(f"The predefined date column '{predefined_date_column}' is not found in the dataset.")
        else:
            date_column = predefined_date_column  # Set the date column automatically
            st.success(f"Date column set to '{date_column}'.")

        # Target column selection
        chosen_target = st.selectbox("Select the target column", options=df.columns, help="Choose the column to forecast.")

        if st.button("Submit"):
            try:
                # Prepare data
                df[date_column] = pd.to_datetime(df[date_column])  # Ensure datetime format

                # Check for duplicate datetime values
                duplicates = df[date_column].duplicated(keep=False)
                if duplicates.any():
                    st.warning(f"Duplicate datetime values found: {duplicates.sum()} duplicates.")
                    df = df.groupby(date_column).mean().reset_index()

                df = df.sort_values(by=date_column)  # Sort by datetime
                df.set_index(date_column, inplace=True)  # Set as index

                # Save to session_state
                st.session_state['prepared_df'] = df
                st.session_state['chosen_target'] = chosen_target
                st.session_state['date_column'] = date_column

                st.success("Data prepared successfully for time series modeling!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error in preparing data: {e}")
    else:
        st.error("No filtered dataset found. Please apply a filter first.")

# Model Comparison section
# Model Comparison section
if choice == "Comparison":
    if 'prepared_df' in st.session_state:
        st.title("Model Comparison")

        # Use prepared data
        df = st.session_state['prepared_df']
        chosen_target = st.session_state.get('chosen_target', None)

        if not chosen_target:
            st.error("Target column not selected. Please go back to 'Target' to set up your data.")
        else:
            st.info(f"Target column: {chosen_target}")

            # Add a reset button to allow re-running the comparison
            if st.button("Reset Model Comparison"):
                st.session_state.pop('best_model', None)
                st.session_state.pop('compare_df', None)
                st.session_state.pop('fine_tuned_model', None)
                st.warning("Model comparison state has been reset. You can now re-run the comparison.")

            # Check if comparison has already been performed
            if 'best_model' not in st.session_state:
                if st.button("Run Model Comparison"):
                    try:
                        with st.spinner("Comparing models, please wait..."):
                            setup(
                                data=df,
                                target=chosen_target,
                                session_id=123,
                                fold=3,
                                verbose=False
                            )

                            best_model = compare_models()
                            compare_df = pull()  # Get the comparison table
                            st.session_state['best_model'] = best_model  # Save the best model
                            st.session_state['compare_df'] = compare_df  # Save the comparison table
                            st.success("Best model determined successfully!")
                            st.dataframe(compare_df)  # Display comparison table
                    except Exception as e:
                        st.error(f"Error during model comparison: {e}")
            else:
                # Display already stored comparison results
                st.success("Model comparison already completed. Below are the results:")
                st.dataframe(st.session_state['compare_df'])

            # Fine-Tune Button
            if 'best_model' in st.session_state:
                if st.button("Fine-Tune Best Model"):
                    try:
                        with st.spinner("Fine-tuning the best model..."):
                            fine_tuned_model = tune_model(st.session_state['best_model'])  # Fine-tune the best model
                            st.session_state['fine_tuned_model'] = fine_tuned_model  # Save fine-tuned model
                            st.success("Model fine-tuned successfully!")
                            st.write("Fine-Tuned Model:", fine_tuned_model)
                    except Exception as e:
                        st.error(f"Error during fine-tuning: {e}")
    else:
        st.error("Data not prepared. Please go to 'Target' to set up your data first.")

# Final Plot Section
if choice == "Final Plot":
    st.title("Final Model Plot")

    # Check if a fine-tuned model exists
    if 'fine_tuned_model' in st.session_state:
        fine_tuned_model = st.session_state['fine_tuned_model']

        # Define the maximum limit for n_sessions
        MAX_SESSIONS = 366  # Adjust based on your use case
        n_sessions = st.number_input(
            "Enter number of future periods to forecast:",
            min_value=1,
            max_value=MAX_SESSIONS,
            value=13,
            step=1,
            help=f"Select a value between 1 and {MAX_SESSIONS}."
        )

        # Generate final plot (e.g., forecast plot)
        try:
            with st.spinner("Generating final plot..."):
                plot_model(
                    fine_tuned_model,
                    plot="forecast",
                    display_format="streamlit",
                    data_kwargs={"fh": n_sessions}  # Number of future periods
                )
                st.success("Final plot generated successfully!")
        except Exception as e:
            st.error(f"Error generating the final plot: {e}")
    else:
        st.warning("No fine-tuned model found. Please fine-tune a model first.")



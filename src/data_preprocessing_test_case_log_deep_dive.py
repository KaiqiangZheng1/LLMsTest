import os
import pandas as pd
import matplotlib.pyplot as plt

#Test case 20240124
def count_eventtemplate_occurrences(df):
    # Group by 'EventTemplate' and count occurrences
    df = df.groupby('EventTemplate', as_index=False).size()
    df.columns = ['EventTemplate', 'Occurrence']
    eventtemplate_counts = df.sort_values(by="Occurrence", ascending=False).reset_index(drop=True)
    # Convert the resulting DataFrame to a list of lists
    list_of_lists = eventtemplate_counts.values.tolist()
    
    return eventtemplate_counts, list_of_lists

def plot_eventtemplates(df, eventtemplate_counts_df, filename, resample_interval='1min', top_n=10):
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('processed_data/log_plots'):
        os.makedirs('processed_data/log_plots')
    
    # Prepare the filename for the plot image
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    # Remove the suffix "_messages_structured"
    base_filename = base_filename.replace('_messages_structured', '')
    plot_filename = os.path.join('processed_data/log_plots', f'{base_filename}.png')
    
    # Convert timestamp column to datetime objects
    df['timestamp'] = pd.to_datetime(df['Time'])
    
    # Set the timestamp as the DataFrame index
    df.set_index('timestamp', inplace=True)
    
    # Determine templates to use based on top_n
    if top_n is not None and top_n > 0:
        # Limit to top N templates by total occurrence
        available_templates = eventtemplate_counts_df.shape[0]
        top_n = min(top_n, available_templates)
        top_templates = eventtemplate_counts_df.nlargest(top_n, 'Occurrence')['EventTemplate']
        title_suffix = f' (Top {top_n})'
    else:
        # Use all templates
        top_templates = eventtemplate_counts_df['EventTemplate']
        title_suffix = ''
    
    df_top = df[df['EventTemplate'].isin(top_templates)]
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # First Plot: Each template occurrence over time (aggregated if resample_interval is provided)
    ax1 = axes[0]
    # Pivot the DataFrame to have templates as columns and timestamps as rows
    pivot_df = df_top.pivot_table(
        index=df_top.index, 
        columns='EventTemplate', 
        values='Occurrence', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Resample the data if resample_interval is provided
    if resample_interval:
        resampled_df = pivot_df.resample(resample_interval).sum()
        interval_label = f' ({resample_interval} Intervals)'
    else:
        resampled_df = pivot_df  # Do not resample
        interval_label = ''
    
    # Cumulative sum over time for each template
    cumulative_df = resampled_df.cumsum()
    
    # Check if cumulative_df is not empty and has more than one unique timestamp
    if not cumulative_df.empty and len(cumulative_df.index.unique()) > 1:
        try:
            # Plot each template's occurrence over time
            cumulative_df.plot(ax=ax1)
            ax1.set_title(f'Templates Occurrence Over Time{title_suffix}{interval_label}')
            ax1.set_xlabel('Timestamp')
            ax1.set_ylabel('Cumulative Occurrence')
            ax1.legend(title='Templates', loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax1.grid(True)
        except Exception as e:
            print(f"Error plotting time series for {filename}: {e}")
            fig.delaxes(ax1)  # Remove the subplot if plotting fails
    else:
        print(f"No sufficient data to plot time series for {filename}")
        fig.delaxes(ax1)  # Remove the subplot if no data
    
    # Second Plot: Templates vs Total Occurrence
    ax2 = axes[1]
    template_counts_top = eventtemplate_counts_df[eventtemplate_counts_df['EventTemplate'].isin(top_templates)]
    
    # Check if template_counts_top is not empty
    if not template_counts_top.empty:
        try:
            template_counts_top.plot(
                kind='bar', 
                x='EventTemplate', 
                y='Occurrence', 
                ax=ax2, 
                legend=False
            )
            ax2.set_title(f'Total Occurrence of Templates{title_suffix}')
            ax2.set_xlabel('Template')
            ax2.set_ylabel('Total Occurrence')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.grid(True)
        except Exception as e:
            print(f"Error plotting bar chart for {filename}: {e}")
            fig.delaxes(ax2)
    else:
        print(f"No data to plot bar chart for {filename}")
        fig.delaxes(ax2)
    # Reduce font sizes
    plt.rcParams.update({'font.size': 6})

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved as {plot_filename}")


def process_all_files_in_folder(folder_path):
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('_messages_structured.csv'):
            print(f"\nProcessing log file: {filename}")
            file_path = os.path.join(folder_path, filename)
            
            # Load the log file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Ensure the required columns exist
            if 'EventTemplate' not in df.columns or 'Time' not in df.columns:
                print(f"File {filename} does not contain required columns. Skipping.")
                continue
            
            # Add 'Occurrence' column for counting purposes
            df['Occurrence'] = 1

            # Use the function to count EventTemplate occurrences
            eventtemplate_counts_df, eventtemplate_counts_list = count_eventtemplate_occurrences(df)
            
            # Plot the templates
            plot_eventtemplates(df, eventtemplate_counts_df, filename, resample_interval=None)  # Adjust as needed for plotting


if __name__ == "__main__":
    # Specify the path to your 'data' folder
    folder_path = r'C:\UofT\Fall 3\ECE1786\Project\Dataset\Test\20240124\pod'  # Replace with your actual folder path

    # Call the function to process all files in the folder
    process_all_files_in_folder(folder_path)

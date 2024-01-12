import subprocess
import time

# Define the output file path
output_file = "ram_usage.csv"

# Function to run Glances and capture RAM usage
def monitor_ram():
    try:
        with open(output_file, "w") as log_file:
            # Start Glances in server mode with RAM module enabled
            process = subprocess.Popen(["glances", "-s", "-r", "--export-csv", "--export-csv-file", output_file], stdout=log_file, stderr=log_file)

            # Monitor RAM usage for a specified duration
            duration = 60  # 60 seconds, adjust as needed
            time.sleep(duration)

            # Stop Glances
            process.terminate()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    monitor_ram()

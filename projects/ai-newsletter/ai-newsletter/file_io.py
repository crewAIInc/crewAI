from datetime import datetime

def save_markdown(task_output):
    today_date = datetime.now().strftime('%Y-%m-%d')
    filename = f"{today_date}.md"
    try:
        # Ensure the output is a string; adjust based on actual method/property
        output = task_output.result() if callable(task_output.result) else str(task_output.result)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(output)
        print(f"Newsletter saved as {filename}")
    except TypeError as e:
        print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

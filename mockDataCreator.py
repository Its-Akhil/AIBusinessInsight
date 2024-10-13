import math
import random
import datetime
import csv

# Initialize variables to maintain the trend
sales_base = 3000
customers_base = 120
average_order_value_base = 100
customer_satisfaction_base = 4.3
time_step = 0  # To simulate time progression


# Function to generate mock data points with a timestamp of yesterday
def generate_mock_data(start_time):
    global time_step

    # Introduce a sine wave trend with some noise for sales and customers
    sales_trend = (
        sales_base + 500 * math.sin(time_step / 10) + random.randint(-100, 100)
    )
    customers_trend = (
        customers_base + 20 * math.sin(time_step / 15) + random.randint(-10, 10)
    )

    # Linear trend with noise for average order value and customer satisfaction
    average_order_value_trend = (
        average_order_value_base + (time_step * 0.05) + random.uniform(-10, 10)
    )
    customer_satisfaction_trend = (
        customer_satisfaction_base + (time_step * 0.01) + random.uniform(-0.1, 0.1)
    )

    # Ensure values stay within realistic bounds
    sales_trend = max(1000, min(sales_trend, 5000))
    customers_trend = max(50, min(customers_trend, 200))
    average_order_value_trend = max(50, min(average_order_value_trend, 200))
    customer_satisfaction_trend = max(3.5, min(customer_satisfaction_trend, 5.0))

    # Calculate the timestamp by adding the time_step (in seconds) to the start_time
    current_time = start_time + datetime.timedelta(seconds=time_step)

    # Generate mock business metrics with trends
    mock_data = {
        "timestamp": current_time.isoformat(),
        "sales": int(sales_trend),
        "customers": int(customers_trend),
        "average_order_value": round(average_order_value_trend, 2),
        "customer_satisfaction": round(customer_satisfaction_trend, 1),
    }

    return mock_data


# Generate 20,000 data points and save to CSV
def generate_csv(filename, num_data_points):
    with open(filename, mode="w", newline="") as file:
        fieldnames = [
            "timestamp",
            "sales",
            "customers",
            "average_order_value",
            "customer_satisfaction",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        global time_step
        # Set the start time to be yesterday at 00:00:00
        start_time = datetime.datetime.now() - datetime.timedelta(days=1)
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        for _ in range(num_data_points):
            data_point = generate_mock_data(start_time)
            writer.writerow(data_point)
            time_step += 1  # Increment time by 1 minute for each data point


if __name__ == "__main__":
    # Generate 20,000 data points and save to 'mock_data.csv'
    generate_csv("mock_data.csv", 20000)
    print("Data generation complete. CSV file 'mock_data.csv' created.")

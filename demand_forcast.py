import itertools
from datetime import datetime, timedelta

inventory_cost = 2

penrose = 'Penrose'
glenfield = 'Glenfield'
city = 'City'

shops = [penrose, glenfield, city]

macbook = 'macbook'
iphone = 'iphone'
shop_stores = [macbook, iphone]
class Consumer:
    def __init__(self, consumer_id, arrival_time, shopping_list,location):
        self.consumer_id = consumer_id
        self.arrival_time = arrival_time
        self.shopping_list = shopping_list
        self.location = location
    def __repr__(self):
        return f"{self.location }Consumer(consumer_id={self.consumer_id}, arrival_time={self.arrival_time} shopping_list={self.shopping_list})"


class ConsumerArrivals:
    def __init__(self):
        self.consumers = {}

    def add_consumer(self, consumer_id, consumer):
        self.consumers[consumer_id] = consumer

    def get_consumer(self, consumer_id):
        return self.consumers.get(consumer_id, None)

    def get_all_consumers(self):
        return list(self.consumers.values())

    def generate_consumers(self):
        # Generate customers for July
        locations = ['Penrose', 'Glenfield', 'City']
        id = 0

        # Generate customers for each location
        for location in locations:
            num_customer_locations = {'Penrose': 1, 'Glenfield': 1, 'City': 6}

            # Generate customers for July
            start_time_july = datetime(2023, 7, 1, 7, 0)
            #end_time_july = datetime(2023, 7, 31, 10, 0)
            time_interval_july = timedelta(hours=1)
            num_customers_july = num_customer_locations[location]

            # Generate customers for August
            #start_time_august = datetime(2023, 8, 1, 7, 0)
            end_time_august = datetime(2023, 8, 27, 10, 0)
            total_days = (end_time_august - start_time_july).days
            #print(total_days)
            time_interval_august = timedelta(hours=1)
            num_customers_august = num_customer_locations[location]

            current_time = start_time_july
            while current_time <= end_time_august:
                # Generate customers only between 7 am and 10 am each day
                if current_time.hour >= 7 and current_time.hour <= 10:
                    if current_time.month == 7:
                        for i in range(num_customers_july):
                            consumer_id = f"July_{id}"
                            shopping_list = {'macbook': 1, 'iphone': 5}  # Example shopping list
                            consumer = Consumer(consumer_id, current_time, shopping_list, location)
                            self.add_consumer(consumer_id, consumer)
                            id += 1

                    if current_time.month == 8:
                        for i in range(num_customers_august):
                            consumer_id = f"August_{id}"
                            shopping_list = {'macbook': 5, 'iphone': 0}  # Example shopping list
                            consumer = Consumer(consumer_id, current_time, shopping_list, location)
                            self.add_consumer(consumer_id, consumer)
                            id += 1

                current_time += time_interval_july

                # Adjust the time interval for August
                if current_time.month == 8:
                    current_time += time_interval_august - time_interval_july


# ShopEnviro class with consumer_arrivals as a class member

# ...
if __name__ == "__main__":
    # Create a ConsumerArrivals object
    consumer_arrivals = ConsumerArrivals()

    # Generate 5 consumers with arrival times starting from 2023-07-01 00:00
    consumer_arrivals.generate_consumers()

    # Get all consumers and print their information
    all_consumers = consumer_arrivals.get_all_consumers()
    for consumer in all_consumers:
        print(consumer)


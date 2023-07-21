from datetime import datetime, timedelta

class Consumer:
    def __init__(self, consumer_id, arrival_time, shopping_list):
        self.consumer_id = consumer_id
        self.arrival_time = arrival_time
        self.shopping_list = shopping_list

    def __repr__(self):
        return f"Consumer(consumer_id={self.consumer_id}, arrival_time={self.arrival_time}, shopping_list={self.shopping_list})"


class ConsumerArrivals:
    def __init__(self):
        self.consumers = {}

    def add_consumer(self, consumer_id, consumer):
        self.consumers[consumer_id] = consumer

    def get_consumer(self, consumer_id):
        return self.consumers.get(consumer_id, None)

    def get_all_consumers(self):
        return list(self.consumers.values())

    def generate_consumers(self, num_consumers):
        start_time = datetime(2023, 7, 1, 0, 0)
        time_interval = timedelta(minutes=10)

        for i in range(num_consumers):
            consumer_id = i + 1
            arrival_time = start_time + i * time_interval
            shopping_list = {'macbook': 1, 'iphone': 2}  # Example shopping list
            consumer = Consumer(consumer_id, arrival_time, shopping_list)
            self.add_consumer(consumer_id, consumer)

# Example Usage:
if __name__ == "__main__":
    # Create a ConsumerArrivals object
    consumer_arrivals = ConsumerArrivals()

    # Generate 5 consumers with arrival times starting from 2023-07-01 00:00
    consumer_arrivals.generate_consumers(5)

    # Get all consumers and print their information
    all_consumers = consumer_arrivals.get_all_consumers()
    for consumer in all_consumers:
        print(consumer)

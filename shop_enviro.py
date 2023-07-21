from SimulationObject import *
from demand_forcast import *
SimSetting.setup(backlogWeek=0, rosterWeekBlockSpan=4, jobdays=10,year = 2023, month = 7, day = 1)



class ShopEnviro(Simulation):

    def __init__(self, isLocalSim=True):
        super().__init__(isLocalSim)
        self.env.process(self.macbook_refill())
        self.env.process(self.iphone_refill())

        self.macbook = simpy.Store(self.env, capacity=2)
        self.iphone = simpy.Store(self.env, capacity=2)
        self.shopping_map = {'macbook': self.macbook, 'iphone': self.iphone}

        self.consumer_arrivals = None
        self.get_data()
    def get_data(self):
        # Create a ConsumerArrivals object
        self.consumer_arrivals:list[Consumer] = ConsumerArrivals()

        # Load data or generate consumers and add them to the ConsumerArrivals object
        # For demonstration purposes, let's add 5 consumers
        num_consumers = 5
        self.consumer_arrivals.generate_consumers(num_consumers)
        for consumer in self.consumer_arrivals.get_all_consumers():
            arrival_sec = (consumer.arrival_time -  SimSetting.start_time_sim).total_seconds()
            print('arrival sec', arrival_sec)
            self.env.process(self.consumer(id=consumer.consumer_id,arrival_time=arrival_sec,
                                           shopping_list=consumer.shopping_list))


    def macbook_refill(self):
        while True:
            yield self.env.timeout(2)

            yield self.macbook.put(f'spam + {self.env.now}')
            print('put macbook item at', self.env.now)

    def iphone_refill(self):
        while True:
            yield self.env.timeout(20)

            yield self.iphone.put(f'spam + {self.env.now}')
            print('put phone item at', self.env.now)

    def consumer(self, id=0,shopping_list={'macbook': 4, 'iphone':3}, arrival_time=10):

        yield self.env.timeout(arrival_time)
        print(id,'requesting spam at', format_time(self.env.now))

        all_requests = []
        for store, value in shopping_list.items():
            for i in range(value):
                all_requests.append(self.shopping_map[store].get())
        yield self.env.all_of(all_requests)

        print(id,'got all at', format_time(self.env.now))


    def run(self, until=2 * day_in_seconds):
        self.env.run(until=until)

shopEnv = ShopEnviro()
shopEnv.run()
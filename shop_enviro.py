import numpy as np

from SimulationObject import *
from demand_forcast import *
from shop_ppo import *
SimSetting.setup(backlogWeek=0, rosterWeekBlockSpan=4, jobdays=10,year = 2023, month = 7, day = 1)



class ShopEnviro(Simulation):

    def __init__(self, isLocalSim=True):
        super().__init__(isLocalSim)
        self.price = {macbook: 5.0, iphone: 2.0}
        self.revenue = 0
        self.shop_revenue = {penrose: 0, city:0, glenfield: 0}
        # Create separate stores for macbook and iphone at each location
        self.macbook_penrose = simpy.Store(self.env, capacity=2000)
        self.iphone_penrose = simpy.Store(self.env, capacity=2000)
        self.macbook_glenfield = simpy.Store(self.env, capacity=2000)
        self.iphone_glenfield = simpy.Store(self.env, capacity=2000)
        self.macbook_city = simpy.Store(self.env, capacity=2000)
        self.iphone_city = simpy.Store(self.env, capacity=2000)

        for i in range(150):
            self.macbook_city.put('init')
            self.iphone_city.put('init')

        self.state = [0,0] # stock level at city
        self.period_reward = 0
        self.inventory_period_cost = 0


        self.shopping_map = {
            penrose: {'macbook': self.macbook_penrose, 'iphone': self.iphone_penrose},
            glenfield: {'macbook': self.macbook_glenfield, 'iphone': self.iphone_glenfield},
            city: {'macbook': self.macbook_city, 'iphone': self.iphone_city}
        }

        self.consumer_arrivals = None
        self.history = []  # Initialize an empty list to store order fulfillment times
        self.get_data()

    def reset(self):
        self.state = np.array([0,0])
        return self.state, 0, False

    def step(self, action):
        done = False
        self.period_reward = 0
        #calculate inventory cost
        total_items = np.sum(self.state)
        self.inventory_period_cost = inventory_cost * total_items
        self.env.process(self.city_macbook_refill(action_space[action][0]))
        self.env.process(self.city_iphone_refill(action_space[action][1]))


        self.env.run(until = self.env.now + 3 * day_in_seconds)
        if self.env.now == 57 * day_in_seconds:

            done = True
            df_history = pd.DataFrame(self.history)

            # Save the DataFrame to a CSV file
            df_history.to_csv('consumer_history.csv', index=False)
        next_state = np.array(self.stock_monitor(shop=city))
        self.state = next_state.copy()
        #print('accumulate reward', self.period_reward)
        return next_state, self.period_reward, done




    def get_data(self):
        # Create a ConsumerArrivals object
        self.consumer_arrivals:ConsumerArrivals = ConsumerArrivals()

        # Load data or generate consumers and add them to the ConsumerArrivals object
        # For demonstration purposes, let's add 5 consumers
        self.consumer_arrivals.generate_consumers()
        for consumer in self.consumer_arrivals.get_all_consumers():
            arrival_sec = (consumer.arrival_time -  SimSetting.start_time_sim).total_seconds()
            location = consumer.location
            self.env.process(self.consumer_arrive(id=consumer.consumer_id, arrival_time=arrival_sec,
                                                  shopping_list=consumer.shopping_list, location=location))

        self.env.process(self.glenfield_macbook_refill())
        self.env.process(self.glenfield_iphone_refill())
        #self.env.process(self.city_iphone_refill())
        #self.env.process(self.city_macbook_refill())
        self.env.process(self.penrose_iphone_refill())
        self.env.process(self.penrose_macbook_refill())

    def glenfield_macbook_refill(self):
        while True:
            yield self.env.timeout(day_in_seconds)

            yield self.macbook_glenfield.put(f'spam + {self.env.now}')
            #print('put macbook item at', self.env.now)

    def glenfield_iphone_refill(self):
        while True:
            yield self.env.timeout(10 * seconds_per_hour)

            yield self.iphone_glenfield.put(f'spam + {self.env.now}')
            #print('put phone item at', self.env.now)

    def city_macbook_refill(self, quantity = 10):

        yield self.env.timeout(1 * day_in_seconds)
        all_put = []
        for q in range(quantity):
            all_put.append(self.macbook_city.put(f'macbook_{q}'))
        yield self.env.all_of(all_put)
        #print('put macbook item at', format_time(self.env.now))
        self.stock_monitor()

    def city_iphone_refill(self,quantity = 10):
        yield self.env.timeout(1 * day_in_seconds)
        all_put = []
        for q in range(quantity):
            all_put.append(self.iphone_city.put(f'macbook_{q}'))
        yield self.env.all_of(all_put)
        #print('put phone item at', self.env.now)


    def penrose_macbook_refill(self):
        while True:
            yield self.env.timeout(500)

            yield self.macbook_penrose.put(f'spam + {self.env.now}')
            #print('put macbook item at', self.env.now)

    def penrose_iphone_refill(self):
        while True:
            yield self.env.timeout(20)

            yield self.iphone_penrose.put(f'spam + {self.env.now}')
            #print('put phone item at', self.env.now)

    def consumer_arrive(self, id=0, shopping_list={'macbook': 4, 'iphone':3}, arrival_time=10, location=None):

        yield self.env.timeout(arrival_time)
        #print(id,'requesting spam at', format_time(self.env.now))
        quit_event = self.env.timeout(500)
        or_events = [quit_event]
        all_requests = []

        revenue = 0
        for store, value in shopping_list.items():
            revenue += self.price[store] * value
            for i in range(value):
                all_requests.append(self.shopping_map[location][store].get())
        or_events.append(all_requests)
        result = yield self.env.any_of([quit_event, self.env.all_of(all_requests)])
        if quit_event not in result:
        #print(id,'got all at', format_time(self.env.now))
            if location == city:
                self.period_reward += revenue
            end_time = self.env.now
            consumer_info = {
                'id': id,
                'location': location,
                'shopping_list': shopping_list,
                'arrival_time': format_time(arrival_time),
                'fulfillment_time': format_time(end_time)
            }
            self.revenue += revenue
            self.shop_revenue[location] += revenue
            self.history.append(consumer_info)
        #self.stock_monitor()

    def stock_monitor(self, shop=None):
        if shop is None:
            for shop in shops:
                for shop_store in shop_stores:
                    pass
                    debug_output(f'number of {shop_store} at {shop} = {len(self.shopping_map[shop][shop_store].items)}')
            #print(f'current revenue = {self.revenue}')
            #print(f'current shop revenue = {self.shop_revenue}')
        else:
            state = [len(self.shopping_map[shop][macbook].items), len(self.shopping_map[shop][iphone].items)]
            return state


    def run(self, until=30 * day_in_seconds):
        self.env.run(until=until)
        # Convert the history list to a DataFrame
        df_history = pd.DataFrame(self.history)

        # Save the DataFrame to a CSV file
        df_history.to_csv('consumer_history.csv', index=False)


#shopEnv.run()


def model():
    # Initialize the observation, episode return and episode length

    env = ShopEnviro()
    print(env.reset())
    observation, episode_return, episode_length = env.reset()[0], 0, 0
    print(observation)
    epochs = 20
    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        #steps_per_epoch = 4000
        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):

            # Get the logits, action, and take one step in the environment
            # print(observation)
            observation = observation.reshape(1, -1)
            logits, action = sample_action(observation)
            # print( env.step(action[0].numpy()))
            observation_new, reward, done = env.step(action[0].numpy())
            # print('new ob',observation_new)
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                env = ShopEnviro()
                observation, episode_return, episode_length = env.reset()[0], 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    observation, episode_return, episode_length = env.reset()[0], 0, 0
    done = False
    while not done:
        observation = observation.reshape(1, -1)
        action = get_optimal_action(observation)
        print('at', observation, 'action', action)
        observation_new, _, done = env.step(action)
        observation = observation_new



model()
done = False
shopEnv = ShopEnviro()
observation, episode_return, episode_length = shopEnv.reset()[0], 0, 0
while not done:
    observation = observation.reshape(1, -1)
    action = get_optimal_action(observation)
    print('at', observation, 'action', action_space[action])
    observation_new, _, done = shopEnv.step(action)
    observation = observation_new
    print(format_time(shopEnv.env.now))
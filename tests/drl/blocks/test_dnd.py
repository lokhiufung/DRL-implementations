import random

import numpy as np
import torch

from drl.blocks.memory.dnd import DifferentiableNeuralDictionary


class TestDND:

    FLOAT_TOLERANCE = 1e-6

    n_actions = 3
    dim = 32
    n_neighbors = 50
    score_threshold = 0.1
    alpha = 1.0

    dnd = DifferentiableNeuralDictionary(
        n_actions=n_actions,
        dim=dim,  # dim of embedding vector
        n_neighbors=n_neighbors,
        score_threshold=score_threshold,
        alpha=alpha,
    )
    
    def test_lookup(self):
        pass

    def test_forward(self):

        self.dnd.reset()

        # random.seed(0)

        keys = []
        values = []
        actions = []

        for _ in range(100):
            key = [random.gauss(0, 1) for _ in range(self.dim)]
            keys.append(key)

            action = random.choice(range(self.n_actions))
            actions.append(action)
            if action == 1:
                values.append(1.0)
            else:
                values.append(0.0)
        
        action_counts = {}
        for action in range(self.n_actions):
            action_counts[action] = actions.count(action)

        try:
            self.dnd.forward(
                key=torch.tensor(keys, dtype=torch.float)
            )
        except AttributeError:
            print('dnd is not ready. no search engine initialized.')
        
        for action, key, value in zip(actions, keys, values):       
            self.dnd.write_to_buffer(
                action,
                torch.tensor(key, dtype=torch.float),
                torch.tensor(value, dtype=torch.float),
            )
        
        self.dnd.write()

        for action in range(self.n_actions):
            if action in action_counts:
                assert self.dnd.dnds[action].search_engine is not None
        
        values_, actions_, indexes, scores = self.dnd.forward(
            key=torch.tensor(keys[0], dtype=torch.float).unsqueeze(0),
            return_all_values=True
        )
        values_np = np.squeeze(values_.detach().cpu().numpy())
        
        # only action == 1 can have non-zero values
        assert values_np.max() == values_np[1]  # assert the highest value of the returned values is the value with action 
        # assert actions_[0].item() == actions[0]
        # assert scores[0][0][0].item() == 0.0  # the closest key MAY NOT be in returned keys

    def test_write(self):

        self.dnd.reset()

        keys = []
        values = []
        actions = []

        for _ in range(100):
            key = [random.gauss(0, 1) for _ in range(self.dim)]
            keys.append(key)
            values.append(1.0)
            actions.append(random.choice(range(self.n_actions)))
        
        action_counts = {}
        for action in range(self.n_actions):
            action_counts[action] = actions.count(action)

        for action, key, value in zip(actions, keys, values):       
            self.dnd.write_to_buffer(
                action,
                torch.tensor(key, dtype=torch.float),
                torch.tensor(value, dtype=torch.float),
            )

        self.dnd.write()

        for action, action_count in action_counts.items():
            assert len(self.dnd.dnds[action].keys) == action_count
            assert len(self.dnd.dnds[action].values) == action_count
            with torch.no_grad():
                total_value = sum(self.dnd.dnds[action].values).item()
                assert total_value == sum([v for a, v in zip(actions, values) if a == action])

    def test_write_to_buffer(self):
        
        self.dnd.reset()

        random.seed(0)

        keys = []
        values = []
        actions = []

        # ensure that the dnds reach maximum capacity
        for action in range(self.n_actions):
            for _ in range(self.dnd.capacity):
                key = [random.gauss(0, 1) for _ in range(self.dim)]
                value = 1.0
                # action = random.choice(range(self.n_actions))

                keys.append(key)
                values.append(value)
                actions.append(action)

                self.dnd.write_to_buffer(
                    action,
                    torch.tensor(key, dtype=torch.float),
                    torch.tensor(value, dtype=torch.float),
                )

        self.dnd.write()
        for action in range(self.n_actions):
            assert len(self.dnd.dnds[action].keys) == len(self.dnd.dnds[action].key_buffer)
            assert len(self.dnd.dnds[action].values) == len(self.dnd.dnds[action].value_buffer) 
            assert len(self.dnd.dnds[action].keys) == self.dnd.capacity
            assert len(self.dnd.dnds[action].values) == self.dnd.capacity


        _, actions_, indexes, _ = self.dnd.lookup(
            key=torch.tensor(keys[100], dtype=torch.float).unsqueeze(0),
            return_all_values=True
        )

        retrieved_indexes = [idx.item() for idx in indexes[0][0]]
        retrieved_action = actions_[0].item()

        latest_tuple = sorted([(idx, ts) for idx, ts in self.dnd.dnds[retrieved_action].last_visit_time.items()], key=lambda x: x[1], reverse=False)[:self.dnd.n_neighbors]
        latest_indexes = [idx for idx, _ in latest_tuple]
        for idx in latest_indexes:
            assert idx in retrieved_indexes  # the retrieved indexes should all be latest
        # print('latest_indexes: ', latest_indexes)
        # print('retrieved_indexes: ', retrieved_indexes)

    def test_update_to_buffer(self):
        self.dnd.reset()

        random.seed(0)

        keys = []
        values = []
        actions = []

        # ensure that the dnds reach maximum capacity
        for action in range(self.n_actions):
            for _ in range(self.dnd.capacity):
                key = [random.gauss(0, 1) for _ in range(self.dim)]
                value = 1.0
                # action = random.choice(range(self.n_actions))

                keys.append(key)
                values.append(value)
                actions.append(action)

                self.dnd.write_to_buffer(
                    action,
                    torch.tensor(key, dtype=torch.float),
                    torch.tensor(value, dtype=torch.float),
                )

        self.dnd.write()

        self.dnd.update_to_buffer(
            actions=[torch.tensor(0, dtype=torch.long)],
            keys=[torch.tensor([keys[0]], dtype=torch.float)],
            values=[torch.tensor(100.0, dtype=torch.float)]
        )
    
    def test_get_max_value(self):
        self.dnd.reset()

        keys = []
        values = []
        actions = []

        for _ in range(100):
            key = [random.gauss(0, 1) for _ in range(self.dim)]
            keys.append(key)
            values.append(1.0)
            actions.append(random.choice(range(self.n_actions)))
        
        action_counts = {}
        for action in range(self.n_actions):
            action_counts[action] = actions.count(action)

        for action, key, value in zip(actions, keys, values):       
            self.dnd.write_to_buffer(
                action,
                torch.tensor(key, dtype=torch.float),
                torch.tensor(value, dtype=torch.float),
            )

        self.dnd.write()

        max_value = self.dnd.get_max_value()
        
        max_value_ = -np.inf
        for action in range(self.n_actions):
            value = max(self.dnd.dnds[action].values).item()
            if value > max_value_:
                max_value_ = value
        
        assert abs(max_value - max_value_) < self.FLOAT_TOLERANCE

    def test_is_ready(self):
        self.dnd.reset()

        assert not self.dnd.is_ready()          

        keys = []
        values = []
        actions = []

        for _ in range(100):
            key = [random.gauss(0, 1) for _ in range(self.dim)]
            keys.append(key)
            values.append(1.0)
            actions.append(random.choice(range(self.n_actions)))
        
        action_counts = {}
        for action in range(self.n_actions):
            action_counts[action] = actions.count(action)

        for action, key, value in zip(actions, keys, values):       
            self.dnd.write_to_buffer(
                action,
                torch.tensor(key, dtype=torch.float),
                torch.tensor(value, dtype=torch.float),
            )

        assert not self.dnd.is_ready()  # write_to_buffer do not mean write to the keys and values parameters

        self.dnd.write()

        assert self.dnd.is_ready()



if __name__ == '__main__':
    tester = TestDND()

    # tester.test_forward()
    tester.test_write_to_buffer()


import gym
import ccxt

class Action(object):
    def __init__(self, type_):
        self.type = type_


class PlaceOrder(Action):
    def __init__(self, type_, price, qty, order_type, side):
        super().__init__(type_)
        self.price = price
        self.qty = qty
        self.order_type = order_type
        self.side = side
    


class CcxtSingleInstrumentTradingEnv(gym.Env):
    def __init__(self, exchange: ccxt.Exchange, symbol: str):
        self.exchange = exchange
        self.symbol = symbol

        self.market = self.exchange.load_markets()

    def _parse_action(self, action):
        return action

    def step(self, action):
        action = self._parse_action(action)

        if action.type == 'place_order':
            if action.order_type == 'limit' and self.side > 0:
                self.exchange.create_limit_buy_order(self.symbol, action.qty, action.price)
            elif action.order_type == 'limit' and self.side < 0:
                self.exchange.create_limit_sell_order(self.symbol, action.qty, action.price)
            elif action.order_type == 'market' and self.side > 0:
                self.exchange.create_market_buy_order(self.symbol, action.qty, action.price)
            elif action.order_type == 'market' and self.side < 0:
                self.exchange.create_market_sell_order(self.symbol, action.qty, action.price)
            else:
                raise ValueError('Value miss-match in PlaceOrder object.')
        else:
            raise ValueError('Only `PlaceOrder` type is supported.')

        market_data = self.exchange.fetch_ticker(self.symbol)


        balance = self.exchange.fetch_total_balance()
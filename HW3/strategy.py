from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, k = 1, sigma = 1, A = 140, gamma = 1, T = 1) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        self.k = k
        self.sigma = sigma
        self.A = A
        self.gamma = gamma
        self.T = T
        
        
    def naive_strategy(self, sim, receive_ts, size, best_bid, best_ask):
        bid_order = sim.place_order(receive_ts, size, 'BID', best_bid)
        ask_order = sim.place_order(receive_ts, size, 'ASK', best_ask)
        return bid_order, ask_order
        
    def as_strategy(self, sim, receive_ts, size, K, s, q, T, sigma, gamma, A):
        r_a = s + (-1 + 2*q)*(-0.5*sigma*sigma*gamma*(T))
        r_b = s + (1 + 2*q)*(-0.5*sigma*sigma*gamma*(T))
        delta_a = r_a - s + np.log(1 + gamma/K)/gamma
        delta_b = s - r_b + np.log(1+gamma/K)/gamma
        lambda_a = A*(np.exp(-K*delta_a))
        lambda_b = A*(np.exp(-K*delta_b))
        bid_order = sim.place_order(receive_ts, size, 'BID', s - delta_b)
        ask_order = sim.place_order(receive_ts, size, 'ASK', s + delta_a)
        return bid_order, ask_order


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf
        q = 0

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    for trade in trades_list:
                        if trade.side == 'BID':
                            q += trade.size
                        else:
                            q -= trade.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                
                #place order
                s = (best_bid + best_ask)/2
                bid_order, ask_order = self.as_strategy(sim, receive_ts, 0.001,
                                                    self.k, s, q, self.T, self.sigma, self.gamma, self.A)
                
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
    
    


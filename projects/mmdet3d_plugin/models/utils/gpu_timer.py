from typing import List, Dict
import collections
import time
import torch

class GPUTimer:
    def __init__(self, activate=False, indent_level=0, print_prefix='') -> None:
        self.activate = activate
        self.indent_level = indent_level
        self.print_prefix = print_prefix

    def __enter__(self):
        if self.activate:
            torch.cuda.synchronize()
            self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.activate:
            torch.cuda.synchronize()
            self.end_time = time.time()
            
            print_str = '\t' * self.indent_level + \
                        self.print_prefix + \
                        f' GPU time: {(self.end_time - self.start_time) * 1000} ms'
            print(print_str)


class GlobalGPUTimer:
    def __init__(self, activate=False, log_interval=50, warmup=200) -> None:
        self.activate = activate
        self.time_count = 0
        self.log_interval = log_interval
        self.warmup = warmup

        self.start_time_info = collections.defaultdict(list)
        self.end_time_info = collections.defaultdict(list)

    def event_start(self, event_name: str):
        assert isinstance(event_name, str)
        if self.activate and self.time_count >= self.warmup:
            torch.cuda.synchronize()
            self.start_time_info[event_name].append(time.perf_counter())

    def event_end(self, event_name: str):
        assert isinstance(event_name, str)

        if self.activate and self.time_count >= self.warmup:
            assert event_name in self.start_time_info, f'Must record the start of {event_name} first!'
            torch.cuda.synchronize()
            self.end_time_info[event_name].append(time.perf_counter())

    def calculate_duration(self):
        duration_info = collections.defaultdict(list)
        
        for event_name in self.start_time_info.keys():
            event_start_time_list = self.start_time_info[event_name]
            event_end_time_list = self.end_time_info[event_name]
            assert len(event_start_time_list) == len(event_end_time_list)

            event_duration_list = [event_end_time_list[i] - event_start_time_list[i] for i in range(len(event_start_time_list))]
            duration_info[event_name] = event_duration_list

        return duration_info

    def set_activate(self, activate=True):
        self.activate = activate
    
    def set_log_interval(self, log_interval=50):
        self.log_interval = log_interval

    def reset(self, time_count=0):
        self.start_time_info.clear()
        self.end_time_info.clear()
        self.reset_time_count(time_count)

    def reset_time_count(self, count=0):
        self.time_count = count

    def update_time_count(self, delta_count=1):
        self.time_count += delta_count

    def parse_duration_info(self, duration_info: dict):
        durations = dict()

        for event_name in duration_info.keys():
            duration_time_list = duration_info[event_name]
            event_classes = event_name.split('/')

            cur_dict = durations
            for cls_rank, event_cls in enumerate(event_classes):
                if event_cls not in cur_dict:
                    cur_dict[event_cls] = dict()
                cur_dict = cur_dict[event_cls]

                if cls_rank == len(event_classes) - 1:
                    cur_dict['sum'] = sum(duration_time_list)
                    cur_dict['mean'] = sum(duration_time_list) / len(duration_time_list)
                    cur_dict['max'] = max(duration_time_list)
                    cur_dict['min'] = min(duration_time_list)
                    cur_dict['len'] = len(duration_time_list)
                    cur_dict['leaf'] = True
                    cur_dict['name'] = event_cls
        
        return durations
    
    def traverse(self, durations: dict, indent_level=0):
        if 'leaf' in durations and durations['leaf']:
            log_str = '\t' * indent_level + \
                        f'mean: {durations["mean"] * 1000}ms, min: {durations["min"] * 1000}ms, max: {durations["max"] * 1000}ms, len: {durations["len"]}'
            print(log_str)
            return durations['mean']
        
        else:
            total = 0
            for k, v in durations.items():
                log_str = '\t' * indent_level + f'{k}:'
                print(log_str)
                total += self.traverse(v, indent_level + 1)
            print('\t' * indent_level + f'total: {total * 1000}ms')
            return total

    def log(self):
        if self.activate and self.time_count > self.warmup and self.time_count % self.log_interval == 0:
            durations = self.parse_duration_info(self.calculate_duration())

            print(f'GlobalGPUTimer (time_count = {self.time_count}, warmup = {self.warmup}):')
            self.traverse(durations, indent_level=1)

GLOBAL_TIMER = GlobalGPUTimer(activate=False, log_interval=50)


if __name__ == '__main__':
    total_epochs = 100
    q = torch.rand([2, 900, 64])
    k = torch.rand([2, 900, 64])
    v = torch.rand([2, 900, 64])

    timer = GlobalGPUTimer(True, log_interval=10)

    with torch.no_grad():
        for e in range(total_epochs):
            
            timer.event_start('attention/mat_mul')
            attn = q @ k.transpose(-2, -1)  # (B * nHead, N, N)
            timer.event_end('attention/mat_mul')
            
            timer.event_start('attention/softmax')
            softmax = torch.softmax(attn, dim=-1)
            timer.event_end('attention/softmax')

            timer.event_start('attention/value')
            value = attn @ v
            timer.event_end('attention/value')

            timer.update_time_count()
            timer.log()


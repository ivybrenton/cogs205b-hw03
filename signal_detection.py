import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class SignalDetection:

    def __init__(self, hits, misses, false_alarms, correct_rejections):
        values = [hits, misses, false_alarms, correct_rejections]
        for v in values:
            if not isinstance(v, (int, float)):
                raise TypeError("All inputs must be numeric.")
            if v < 0:
                raise ValueError("Counts cannot be negative.")
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        denom = self.hits + self.misses
        if denom == 0:
            return 0.0
        return self.hits / denom

    def false_alarm_rate(self):
        denom = self.false_alarms + self.correct_rejections
        if denom == 0:
            return 0.0
        return self.false_alarms / denom
    
    def d_prime(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        H = np.clip(H, 1e-5, 1 - 1e-5)
        FA = np.clip(FA, 1e-5, 1 - 1e-5)
        return norm.ppf(H) - norm.ppf(FA)
    
    def criterion(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()
        H = np.clip(H, 1e-5, 1 - 1e-5)
        FA = np.clip(FA, 1e-5, 1 - 1e-5)
        return -0.5 * (norm.ppf(H) + norm.ppf(FA))
    
    def __add__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Can only add another SignalDetection object.")
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )
    
    def __sub__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Can only subtract another SignalDetection object.")
        return SignalDetection(
            self.hits - other.hits,
            self.misses - other.misses,
            self.false_alarms - other.false_alarms,
            self.correct_rejections - other.correct_rejections
        )
    
    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise TypeError("Factor must be numeric.")
        return SignalDetection(
            self.hits * factor,
            self.misses * factor,
            self.false_alarms * factor,
            self.correct_rejections * factor
        )
    
    @staticmethod
    def plot_roc(sdt_list):
        hit_rates = []
        false_alarm_rates = []
        for sdt in sdt_list:
            if not isinstance(sdt, SignalDetection):
                raise TypeError("All elements must be SignalDetection objects.")
            hit_rates.append(sdt.hit_rate())
            false_alarm_rates.append(sdt.false_alarm_rate())
        hit_rates = [0] + hit_rates + [1]
        false_alarm_rates = [0] + false_alarm_rates + [1]
        points = list(zip(hit_rates, false_alarm_rates))
        points = sorted(points, key=lambda x: x[0])
        hit_rates, false_alarm_rates = zip(*points)
        fig, ax = plt.subplots()
        ax.plot(hit_rates, false_alarm_rates, marker='o', color='#ff69b4')
        ax.set_xlabel("Hit Rate")
        ax.set_ylabel("False Alarm Rate")
        ax.set_title("ROC Curve")
        return fig, ax
    
    
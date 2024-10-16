import geopy.distance
import numpy as np
from scipy.optimize import least_squares

# given a velocity profile, extrapolate some points using the 5 previous ones
def extrapolate_velocities_linear(depths, velocities, to_extrapolate):
    slope = (velocities[-3, :, :] - velocities[-1, :, :]) / (depths[-3] - depths[-1])
    extrapolated = velocities[-1, :, :] + slope * (to_extrapolate - depths[-1])[:, np.newaxis, np.newaxis]
    return extrapolated


# abstract class of a SoundModel, defining what a SoundModel should do
class SoundModel():
    # return the time, in s, that a sound emitted at one of the positions would require to reach the other one
    def get_sound_travel_time(self, pos1, pos2, date=None):
        return None

    def localize_common_source(self, sensors_positions, detection_times, x_min=-90, y_min=-180, x_max=90,
                             y_max=180, initial_pos=None):

        reference_date = detection_times[0]  # to determine the part of the year concerned
        min_date = np.argmin(detection_times)
        detection_times = np.array([(d-detection_times[min_date]).total_seconds() for d in detection_times])
        sensors_positions = np.array(sensors_positions)

        def f(x):
            time_s = x[0]
            pos = x[1:]
            array = np.zeros(len(sensors_positions))
            for i in range(len(sensors_positions)):
                try:
                    arrival = time_s + self.get_sound_travel_time(pos, sensors_positions[i], reference_date)
                    array[i] = np.abs(arrival - detection_times[i])
                except:
                    array[i] = np.inf
            return array

        x0 = [0, *(initial_pos if initial_pos is not None else np.mean(sensors_positions, axis=0))]
        x0[0] = -self.get_sound_travel_time(x0[1:], sensors_positions[min_date], reference_date)
        try:
            res = least_squares(f, x0, bounds=([-np.inf, x_min, y_min], [0, x_max, y_max]))
        except:
            # absurd detection times can lead to an error during the least squares
            res = [-1, -1]


        return res

# sound model of homogeneous sound velocity
class HomogeneousSoundModel(SoundModel):
    # sound speed in m/s
    def __init__(self, sound_speed=1480):
        super().__init__()
        self.sound_speed = sound_speed

    def get_sound_travel_time(self, pos1, pos2, date=None):
        distance = geopy.distance.geodesic(pos1[:2], pos2[:2]).m
        return distance / self.sound_speed

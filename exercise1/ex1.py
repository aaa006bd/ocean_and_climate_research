import numpy as np
import pandas as pd


class Model:
    def model_equation(self, output_at_t, t):
        pass


class EnergyBalanceModel(Model):
    def __init__(self):
        self.initial_value = 10

        self.thermal_coupling_constant = 9.96e6
        self.emissivity = 0.62
        self.boltzmann_constant = 5.67e-8
        self.solar_constant = 1367
        self.albedo = 0.3

        self.c1 = 1 / (4 * self.thermal_coupling_constant)
        self.c2 = (self.boltzmann_constant * self.emissivity) / self.thermal_coupling_constant

    def model_equation(self, output_at_t, t):
        output_at_t_plus_1 = (self.c1 * self.solar_constant * (1 - self.albedo)) - (self.c2 * output_at_t ** 4)
        return output_at_t_plus_1


class ExplicitEulerMethodSimulation:
    def __init__(self,s_time,e_time,steps, model):
        self.simulation_output = pd.DataFrame()

        self.start_time = s_time
        step = steps
        end_time = e_time
        self.time_step = (end_time-s_time) / step
        self.time_array = np.arange(self.start_time, end_time, self.time_step)
        self.temp_array = np.empty((len(self.time_array)))
        self.simulation_output['time'] = self.time_array
        self.simulation_output = self.simulation_output.set_index(['time'])

        self.model = model if isinstance(model,Model) else None

    def simulation_step(self, temp, time):
        new_temp = temp + (self.time_step * self.model.model_equation(temp, time))
        return new_temp

    def run_simulation(self):
        if self.model is not None:
            index = 0
            temp_initial = self.model.initial_value
            self.temp_array[index] = temp_initial
            for time in self.time_array[1:]:
                prev_value = self.temp_array[index]
                index += 1
                self.temp_array[index] = self.simulation_step(prev_value, time)
            self.simulation_output['temperature'] = self.temp_array
        else:
            print('invalid model were given for simulation')

    def output_csv(self):
        self.simulation_output.to_csv('output.csv')


simulation_model = ExplicitEulerMethodSimulation(s_time=0, e_time=1000, steps=1000, model=EnergyBalanceModel())
simulation_model.run_simulation()
print(simulation_model.simulation_output.head(5))

simulation_model.output_csv()
import numpy as np
import pandas as pd


class Model:
    def model_equation(self, output_at_t, t):
        pass

    def get_initial_value(self):
        pass


class EnergyBalanceModel(Model):
    def __init__(self, init_val_inC):
        self.kelvin_constant = 273.15
        self.initial_value = init_val_inC + self.kelvin_constant

        self.thermal_coupling_constant = 9.96e6
        self.emissivity = 0.62
        self.boltzmann_constant = 5.67e-8
        self.solar_constant = 1367
        self.albedo = 0.3

        self.c1 = 1 / (4 * self.thermal_coupling_constant)
        self.c2 = (self.boltzmann_constant * self.emissivity) / self.thermal_coupling_constant

    def model_equation(self, output_at_t, t):
        model_output = (self.c1 * self.solar_constant * (1 - self.albedo)) - (self.c2 * output_at_t ** 4)
        return model_output

    def get_initial_value(self):
        return self.initial_value


class ExplicitEulerMethodSimulation:
    def __init__(self,s_time,e_time,steps, model):
        self.simulation_output = pd.DataFrame()

        self.start_time = s_time
        step = steps
        end_time = e_time
        self.time_step = (end_time-s_time) / step
        self.time_array = np.arange(self.start_time, end_time, self.time_step)
        self.output_array = np.empty((len(self.time_array)))
        self.simulation_output['time'] = self.time_array
        self.simulation_output = self.simulation_output.set_index(['time'])

        self.model = model if isinstance(model, Model) else None

    def simulation_step(self, prev_val, time):
        new_val = prev_val + (self.time_step * self.model.model_equation(prev_val, time))
        return new_val

    def run_simulation(self):
        if self.model is not None:
            index = 0
            temp_initial = self.model.get_initial_value()
            self.output_array[index] = temp_initial
            for time in self.time_array[1:]:
                prev_value = self.output_array[index]
                index += 1
                self.output_array[index] = self.simulation_step(prev_value, time)
            self.simulation_output['output'] = self.output_array

        else:
            print('invalid model were given for simulation')

    def output_csv(self):
        self.simulation_output.to_csv('output-init-{}.csv'.format(str(self.simulation_output['output'][0])))


simulator = ExplicitEulerMethodSimulation(s_time=0, e_time=60*60*24*100, steps=60*60*24, model=EnergyBalanceModel(init_val_inC=15))
simulator.run_simulation()
print(simulator.simulation_output.head(5))

simulator.output_csv()
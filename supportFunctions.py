import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(x, y, reg):
    y_pred = reg.predict(x)
    r2 = r2_score(y, y_pred)

    plt.plot(x, y_pred, color = "b", label="Estimated Values")
    plt.scatter(x, y, color = "r", marker = "o", s = 30, label='Simulated Values')
    plt.xlabel('Temperature [°C]')
    plt.ylabel('Vout [V]')
    plt.legend()
    plt.grid()
    
    return y_pred, r2
 
def temperatureSensorRegression():
    df = pd.read_excel('vout vs temp.xlsx')

    temperature = np.array([df['temperature'][:]]).reshape((-1, 1))
    voltage = np.array(df['V(vout)'][:])
 
    reg = LinearRegression().fit(temperature, voltage)
    estimation, r2 = evaluate_model(temperature, voltage, reg)
 
    # Printing function in the following format:
    #   V(T) = m*T + b
    numDigits = 8
    coeffs = reg.intercept_, reg.coef_[0]

    print(f"\nVoltage in terms of the temperature:\n\n \
        V(T) = {round(coeffs[1], numDigits)} * T + {round(coeffs[0], numDigits)}", end = '')
        
    print(f'\t(R^2 Score = {round(r2, numDigits)})\n')
    plt.show()

def pt100():
    # Pt100 Platinum Temperature Sensor coefficients.
    A = 3.9083*10**-3
    B = -5.775*10**-7
    Ro = 100

    # Defining lowest and highest temperature measured by the system.
    Tmin = 0
    Tmax = 40

    T = np.array([Tmin, Tmax])
    Rpt = Ro*(1+A*T+B*T**2)

    i = 5*10**-3

    Vmin = i*Rpt[0]
    Vmax = i*Rpt[1]

    print(f'Delta V = {Vmax-Vmin}')

    safetyMargin = 10 / 100
    desiredVoltageVariation = 3.3*(1-safetyMargin)
    gain = desiredVoltageVariation / (Vmax-Vmin)

    print(f'Needed Gain = {gain}')

    lowestVoltage = gain*Vmin
    print(f'Lowest value assumed by V(out) = {lowestVoltage}')

if __name__ == '__main__':
    temperatureSensorRegression()
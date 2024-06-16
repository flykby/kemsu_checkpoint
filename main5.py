import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 2*x**2 - 4*x + 5

def bisection_method(f, a, b, epsilon):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    while (b - a) / 2.0 > epsilon:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2.0

def iteration_method(f, x0, epsilon, max_iter=1000):
    for i in range(max_iter):
        x1 = f(x0)
        if abs(x1 - x0) < epsilon:
            return x1
        x0 = x1
    print("Iteration method did not converge")
    return None

def secant_method(f, x0, x1, epsilon, max_iter=1000):
    for i in range(max_iter):
        if abs(f(x1) - f(x0)) < epsilon:
            print("Divide by zero error in secant method")
            return None
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < epsilon:
            return x2
        x0, x1 = x1, x2
    print("Secant method did not converge")
    return None

def newton_method(f, df, x0, epsilon, max_iter=100):
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < epsilon:
            return x1
        x0 = x1
    print("Newton's method did not converge")
    return None

# Примерные производные для методов
def df(x):
    return 3*x**2 - 4*x - 4

# Примерные функции для метода итераций и метода хорд
def g1(x):
    return (5/x)**0.5

def g2(x):
    return (3*x)**0.5

# Устанавливаем начальные значения
epsilon = 0.1

# Интервал для метода дихотомии
a, b = 0, 3

# Начальные приближения для других методов
x0 = 10
x1 = 30

# Поиск корней различными методами
root_bisection = bisection_method(f, a, b, epsilon)
print(f"Bisection method root: {root_bisection}")

root_iteration = iteration_method(g1, x0, epsilon)
print(f"Iteration method root: {root_iteration}")

root_secant = secant_method(f, x0, x1, epsilon)
print(f"Secant method root: {root_secant}")

root_newton = newton_method(f, df, x0, epsilon)
print(f"Newton's method root: {root_newton}")

# Построение графика функции
x = np.linspace(-2, 4, 400)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="y = x^3 - 2x^2 - 4x + 5")
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title("График функции")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

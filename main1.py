import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 2*x**2 - 4*x + 5

def df(x):
    return 3*x**2 - 4*x - 4


def bisection_method(a, b, epsilon, max_iterations=100):
    if f(a) * f(b) >= 0:
        raise ValueError("Метод дихотомии не может быть применен, так как f(a) * f(b) >= 0")
    
    iterations = 0
    while (b - a) / 2 > epsilon:
        c = (a + b) / 2.0
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        iterations += 1
        if iterations >= max_iterations:
            raise ValueError("Достигнуто максимальное количество итераций")
    
    return (a + b) / 2.0


def iterative_method(initial_guess, epsilon, max_iterations=100):
    x = initial_guess
    iterations = 0
    while True:
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
        
        iterations += 1
        if iterations >= max_iterations:
            raise ValueError("Достигнуто максимальное количество итераций")
    
    return x


def secant_method(a, b, epsilon, max_iterations=100):
    iterations = 0
    while abs(b - a) > epsilon:
        a, b = b, b - (b - a) * f(b) / (f(b) - f(a))
        
        iterations += 1
        if iterations >= max_iterations:
            raise ValueError("Достигнуто максимальное количество итераций")
    
    return b


def newton_method(initial_guess, epsilon, max_iterations=100):
    x = initial_guess
    iterations = 0
    while True:
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
        
        iterations += 1
        if iterations >= max_iterations:
            raise ValueError("Достигнуто максимальное количество итераций")
    
    return x


def main():
    # Начальные значения и точность
    a, b = -5, 5
    epsilon = 0.1

    # Метод дихотомии
    root_bisection = bisection_method(a, b, epsilon)
    print(f"Метод дихотомии: корень приближенный = {root_bisection:.6f}")

    # Метод итераций
    initial_guess_iterative = 1.5
    root_iterative = iterative_method(initial_guess_iterative, epsilon)
    print(f"Метод итераций: корень приближенный = {root_iterative:.6f}")

    # Метод хорд
    root_secant = secant_method(a, b, epsilon)
    print(f"Метод хорд: корень приближенный = {root_secant:.6f}")

    # Метод Ньютона
    initial_guess_newton = 10.0
    root_newton = newton_method(initial_guess_newton, epsilon)
    print(f"Метод Ньютона: корень приближенный = {root_newton:.6f}")

    x_vals = np.linspace(-5, 5, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='$y = x^3 - 2x^2 - 4x + 5$', color='b')
    plt.title('График функции $y = x^3 - 2x^2 - 4x + 5$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
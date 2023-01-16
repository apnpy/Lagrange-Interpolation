import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline

# Собственная функция для интерполяции Лагранжа
def Lagrange (Lx, Ly):
    x=sp.symbols('x')
    if  len(Lx)!= len(Ly):
        return 1
    y=0
    for k in range ( len(Lx) ):
        t=1
        for j in range ( len(Lx) ):
            if j != k:
                t=t* ( (x-Lx[j]) /(Lx[k]-Lx[j]) )
        y+= t*Ly[k]
    return y


x = [0, 0.5, 1, 1.5, 2, 3]
y = [0, 1, 0, -1, 0, 1]
itog = sp.expand(Lagrange(x,y))

print('Своя функция для нахождения интерполяции методом Лагранжа:')
print('f(x) =',itog)

print('\nРешение встроенной функцией:')
func = lagrange(x, y)
print(func)

# Решение уравнений в точках
def solving_equations(x_new,itog=itog):
    values = []
    for j in range(len(x_new)):
        x = x_new[j]
        eq = round(eval(str(itog)),3)
        values.append(eq)
    return values

# Проверка правильности нахождения функции:
print('\nПроверка правильности нахождения функции:')
values_first = solving_equations(x)
print(values_first,'\nСовпадает со значениями в таблице - функция найдена верно.')

# Средние точки
x2 = [(x[i] + x[i+1]) / 2 for i in range(len(x) - 1)]
print(f'\nСредние точки:',x2)

values = solving_equations(x2)
print('\nЗначения функции в точках:',values)

# Сравнение точных и средних значений в точке
srav = []

for i in range(len(x)-1):
    if values[i] > x2[i]:
        srav.append(f'{i+1}) точное значение > значение в средней точке ({round(values[i], 3)} > {x2[i]})')
    elif values[i] < x2[i]:
        srav.append(f'{i+1}) точное значение < значение в средней точке ({round(values[i], 3)} < {x2[i]})')
    else:
        srav.append(f'{i+1}) точное значение = значение в средней точке ({round(values[i], 3)} = {x2[i]})')
[print(srav[i]) for i in range(len(srav))]

#Построение графиков
def fx(i):
    if i == 1:
        return np.log(1 + x**2)
    elif i == 2:
        return 5 * np.exp(-x**2)
    elif i == 3:
        return np.cosh(x)
    elif i == 4:
        return 2 * np.cos(x)
    elif i == 5:
        return 1/(x**2+1)
    elif i == 6:
        return np.log(x)
    elif i == 7:
        return 1/(1+25*x**2)
    elif i == 8:
        return x**4 + 4*x**3 +1
    elif i == 9:
        return np.sinh(x)
    elif i == 10:
        return 3 * np.arctan(x)

names = ['ln(1 + x^2)','5 * exp(-x^2)','ch(x)','2cos(x)','1/(x^2+1)',
         'ln(x)','1/(1+25x^2)','x^4+4x^3+1','sh(x)','3arctg(x)']
x_new = np.arange(-1, 3.01, 0.01)
x = np.arange(-3, 3.01, 0.01)

ask = input('Отобразить графики в отдельных окнах?да/нет\n')

if ask == 'да':
    for k in range(1,11):

        plt.figure(figsize = (8,8))
        plt.plot(x, fx(k),label = names[k-1])
        plt.plot(x_new, func(x_new), label = [itog])
        plt.title('Построение графиков')
        plt.grid()
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()

ask2 = input('Отобразить графики в одном окне?да/нет\n')
# Графики в одном окне
if ask2 == 'да':
    plt.figure(figsize=(10, 10))

    plt.subplot(4, 3, 1)
    plt.plot(x, fx(1))
    plt.grid()
    plt.title(names[0])

    plt.subplot(4, 3, 2)
    plt.plot(x, fx(2))
    plt.grid()
    plt.title(names[1])

    plt.subplot(4, 3, 3)
    plt.plot(x, fx(3))
    plt.grid()
    plt.title(names[2])

    plt.subplot(4, 3, 4)
    plt.plot(x, fx(4))
    plt.grid()
    plt.title(names[3])

    plt.subplot(4, 3, 5)
    plt.plot(x, fx(5))
    plt.grid()
    plt.title(names[4])

    plt.subplot(4, 3, 6)
    plt.plot(x, fx(6))
    plt.grid()
    plt.title(names[5])

    plt.subplot(4, 3, 7)
    plt.plot(x, fx(7))
    plt.grid()
    plt.title(names[6])

    plt.subplot(4, 3, 8)
    plt.plot(x, fx(8))
    plt.grid()
    plt.title(names[7])

    plt.subplot(4, 3, 9)
    plt.plot(x, fx(9))
    plt.grid()
    plt.title(names[8])

    plt.subplot(4, 3, 11)
    plt.plot(x, fx(10))
    plt.grid()
    plt.title(names[9])

    plt.show()


# Сравнение значения в заданной точке со значением функции y = sin(pi*x)
ask3 = input('Сравнить значения в заданной точке со значением функии?(да/нет)\n')
if ask3 == 'да':
    print('\nСравнение значения в заданной точке со значением функции y = sin(pi*x)\n')

    input_x = float(input('Введите точку x:\n'))
    x = input_x
    func_point = round(eval(str(itog)),4)
    print('Значение функции в заданной точке:',func_point)

    func_zadan = round(np.sin(np.pi*input_x),4)
    print('Значение y = sin(pi*x) в точке',func_zadan)

    if func_point > func_zadan:
        print('Значение функции в заданной точке > Значения y = sin(pi*x)) в точке')
    elif func_point < func_zadan:
        print('Значение функции в заданной точке < Значения y = sin(pi*x)) в точке')
    else:
        print('Значение функции в заданной точке = Значения y = sin(pi*x)) в точке')

#Кубические сплайны
ask4 = input('Вывести кубические сплайны?(да/нет)\n')
if ask4 == 'да':
    for i in range(1, 11):
        if i == 6:
            x = np.arange(0.01, 3.01, 0.01)
        else:
            x = np.arange(-3, 3.01, 0.01)
        y_new1 = fx(i)
        f = CubicSpline(x, y_new1, bc_type = 'natural')
        x_new1 = np.linspace(-3, 3, 100)
        fx_new1 = f(x_new1)
        plt.figure(figsize=(10, 8))
        plt.plot(x_new1, fx_new1, 'g', label=names[i - 1])
        plt.title(f'График {i}. Интерполирование кубическими сплайнами функции: {names[i-1]}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
    plt.show()



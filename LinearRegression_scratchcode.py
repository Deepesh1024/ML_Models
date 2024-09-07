import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('sm.csv')

#visualization
# plt.scatter(data.time_study, data.Marks)
# plt.show()

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].time_study
        y = points.iloc[i].Marks
        total_error += (y- (m*x+b))**2
    return total_error/float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].time_study
        y = points.iloc[i].Marks

        m_gradient += -(2/n)*x*(y-(m_now*x + b_now))
        b_gradient += -(2/n)*(y-(m_now*x + b_now))

        m = m_now - m_gradient*L
        b = b_now - b_gradient*L

    return m,b

def slr(time_study):
    return m*time_study+ b
m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i%50 ==0:
        print(f"Epoch:{i}, Loss: {loss_function(m, b, data)}")
    m,b  = gradient_descent(m,b,data, L)

print(f"Final slope (m): {m}, Intercept (b): {b}")

study_time = float(input("Enter the study time : "))
print(slr(study_time))


plt.scatter(data.time_study, data.Marks, color = "black")
plt.plot(list(range(0,10)), [m*x +b for x in range(0,10)], color = "red")
plt.show()

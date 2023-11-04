import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 定义参数
xF = 0.4
q = 1
R = 1.875
xD = 0.9
alpha = 2.47
F = 100000
Eta_D = 0.9
D = F * xF * xD
W = F - D
L = R * D
xW = (F * xF - D * xD) / W


# 定义平衡曲线
def equilibrium_line(x, alpha):
    return alpha * x / (1 + (alpha - 1) * x)


# 定义精流段操作线
def rectifying_line(x, R, xD):
    return R / (R + 1) * x + xD / (R + 1)


# 绘制q线
def q_line(x, q, xF):
    if q == 1:
        # 当q=1时，馈线是垂直的
        return x
    else:
        return q / (q - 1) * x - xF / (q - 1)


# 绘制提馏段线
def stripping_line(x, q, F, xW, W, L):
    y_m = (L + q * F) / (L + q * F - W) * x - W / (L + q * F - W) * xW
    return y_m


# 平衡曲线的反解
def inverse_equilibrium_line(y, alpha):
    return y / (alpha - y * (alpha - 1))


# 理论塔板计算并绘制步骤
def calculate_theoretical_plates_and_plot_steps(xF, q, R, xD, xW, alpha, F, W, L, max_iter=1000):
    x = xD
    y = R / (R + 1) * x + xD / (R + 1)
    count = 0

    # 迭代计算
    while x > xW and count < max_iter:
        old_x = x
        old_y = y

        if x > xF:
            y = rectifying_line(x, R, xD)
            x = inverse_equilibrium_line(y, alpha)
        else:
            y = stripping_line(x, q, F, xW, W, L)
            x = inverse_equilibrium_line(y, alpha)

        # 绘制从旧x值到新x值的线段
        plt.plot([old_x, x], [y, y], 'r')
        plt.plot([old_x, old_x], [old_y, y], 'r')
        plt.plot([xW, xW],[stripping_line(xW, q, F, xW, W, L),equilibrium_line(xW, alpha)],'b')

        # 确保x值在有效范围内
        x = max(min(x, 1), 0)
        count += 1

    return count if count < max_iter else "Failed to converge"

# 创建Streamlit应用程序并定义布局
st.title("数据可视化示例")

xF = st.slider("xF", 0.0, 1.0, 0.4)
q = st.slider("q", 1, 10, 1)
R = st.slider("R", 0.0, 10.0, 1.875)
xD = st.slider("xD", 0.0, 1.0, 0.9)
alpha = st.slider("alpha", 0.0, 10.0, 2.47)
F = st.slider("F", 0, 100000, 100000)

D = F * xF * xD
W = F - D
L = R * D
xW = (F * xF - D * xD) / W

# 创建x值
x_range = np.linspace(0, 1, 100)

# 计算y值
y_equilibrium = equilibrium_line(x_range, alpha)
y_rectifying = rectifying_line(x_range, R, xD)
y_q = q_line(x_range, q, xF)
y_stripping = stripping_line(x_range, q, F, xW, W, L)

# 创建图形
fig, ax = plt.subplots()

# 绘制平衡曲线
ax.plot(x_range, y_equilibrium, label='Equilibrium line')

# 绘制精流段操作线
ax.plot(x_range, y_rectifying, label='Rectifying line')

# q=1，绘制垂直的q线
ax.axvline(xF, color='green', linestyle='--', label='q_Line')

# 绘制提馏段操作线
ax.plot(x_range, y_stripping, label='stripping_Line')

st.pyplot(fig)

st.write("理论塔板数为", calculate_theoretical_plates_and_plot_steps(xF, q, R, xD, xW, alpha, F, W, L), "（不包括再沸器）")

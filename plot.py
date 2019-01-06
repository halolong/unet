import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
x1 = [0.5, 1, 1.5, 2]
y = [2, 4, 6, 8]
length = range(1, len(x) + 1)

plt.figure(num=1)
plt.plot(length, x, 'bo', label='test1')
plt.plot(length, x1, 'b', label='test2')
plt.legend()
plt.savefig('test.png')


plt.figure(num=2)
plt.plot(length, y, 'b', label='test1')
plt.plot(length, x1, 'bo', label='test2')
plt.legend()
plt.savefig('test1.png')

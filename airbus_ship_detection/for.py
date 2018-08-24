a = [i + 5 for i in range(10)]
print(a)

b = [i * 2 for i in range(10)]
print(b)

c = [i+4 for i in range(0, 10) if i%2 == 0]
print(c)

d = [i+4 for i in range(0, 10) if i%2 == 1]
print(d)

e = [i*j for j in range(2, 10) for i in range(2, 3)]
print(e)

starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

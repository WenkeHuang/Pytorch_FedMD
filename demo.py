import matplotlib.pyplot as plt
demo1 = [1,1,2,3,4]
demo2 = [1,1,5,3,4]
demo = []
demo.append(demo1)
demo.append(demo2)

plt.figure()
for i,val in enumerate(demo):
    plt.plot(range(len(val)),val)
plt.show()
plt.savefig()
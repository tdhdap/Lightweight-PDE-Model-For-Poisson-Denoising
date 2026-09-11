import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('image.png',cv2.IMREAD_GRAYSCALE)

u=img.astype(float)/255.0

a=float(input("Enter a: "))
b=float(input("Enter b: "))

h=1
dt=0.1         
alpha=dt/(h**2)
total_steps=50   

initial_u=u.copy()

for step in range(total_steps):
    u[1:-1,1:-1]=u[1:-1,1:-1]+alpha*(
        a*(u[2:,1:-1]+u[:-2,1:-1]-2*u[1:-1,1:-1])+
        b*(u[1:-1,2:]+u[1:-1,:-2]-2*u[1:-1,1:-1])
    )

    u[0,:]=u[1,:]
    u[-1,:]=u[-2,:]
    u[:,0]=u[:,1]
    u[:,-1]=u[:,-2]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(initial_u, cmap='gray', vmin=0, vmax=1)
plt.title("Original Input")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(u, cmap='gray', vmin=0, vmax=1)
plt.title(f"After {total_steps} Steps (Neumann)")
plt.axis('off')

plt.show()